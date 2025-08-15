using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json;

public class Qwen2Tokenizer
{
    private readonly Dictionary<string, int> _encoder;
    private readonly Dictionary<int, string> _decoder;
    private readonly Dictionary<byte, char> _byteEncoder;
    private readonly Dictionary<char, byte> _byteDecoder;
    private readonly Dictionary<(string, string), int> _bpeRanks;
    private readonly Dictionary<string, List<string>> _bpeCache = new Dictionary<string, List<string>>();
    
    private readonly HashSet<string> _specialTokens;
    private readonly Regex _specialTokensRegex;
    private readonly Regex _pretokenizeRegex;

    public int EosTokenId { get; }
    public int PadTokenId { get; }
    public int UnkTokenId { get; }

    private const string PretokenizePattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    public Qwen2Tokenizer(string vocabJsonContent, string mergesTxtContent, string tokenizerConfigJsonContent)
    {
        _encoder = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabJsonContent);
        _decoder = _encoder.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        
        var tokenizerConfig = JsonConvert.DeserializeObject<TokenizerConfig>(tokenizerConfigJsonContent);
        
        if (tokenizerConfig?.AddedTokensDecoder != null)
        {
            foreach (var kvp in tokenizerConfig.AddedTokensDecoder)
            {
                int tokenId = int.Parse(kvp.Key);
                string tokenContent = kvp.Value.Content;
                if (!_encoder.ContainsKey(tokenContent))
                {
                    _encoder[tokenContent] = tokenId;
                    _decoder[tokenId] = tokenContent;
                }
            }
        }

        _specialTokens = new HashSet<string>();
        if (tokenizerConfig?.AddedTokensDecoder != null)
        {
            foreach (var tokenDef in tokenizerConfig.AddedTokensDecoder.Values)
            {
                if (tokenDef.Special)
                {
                    _specialTokens.Add(tokenDef.Content);
                }
            }
        }
        
        if (_specialTokens.Any())
        {
            var escapedTokens = _specialTokens.Select(Regex.Escape);
            _specialTokensRegex = new Regex($"({string.Join("|", escapedTokens)})", RegexOptions.Compiled);
        }
        else
        {
            _specialTokensRegex = new Regex("(?!)", RegexOptions.Compiled);
        }

        EosTokenId = _encoder[tokenizerConfig.EosToken];
        PadTokenId = _encoder[tokenizerConfig.PadToken];
        
        UnkTokenId = tokenizerConfig.UnkToken != null && _encoder.ContainsKey(tokenizerConfig.UnkToken) 
            ? _encoder[tokenizerConfig.UnkToken] 
            : _encoder["<|endoftext|>"];
        
        _bpeRanks = LoadMergesFromString(mergesTxtContent);

        (_byteEncoder, _byteDecoder) = BuildByteToUnicodeMap();
        
        _pretokenizeRegex = new Regex(PretokenizePattern, RegexOptions.Compiled);
    }
    
    public List<int> Encode(string text)
    {
        text = text.Normalize(NormalizationForm.FormC);
        var tokenIds = new List<int>();
        string[] parts = _specialTokensRegex.Split(text);
        foreach (string part in parts)
        {
            if (string.IsNullOrEmpty(part)) continue;
            if (_specialTokens.Contains(part))
            {
                tokenIds.Add(_encoder[part]);
            }
            else
            {
                var matches = _pretokenizeRegex.Matches(part);
                foreach (Match match in matches)
                {
                    var builder = new StringBuilder();
                    foreach (byte b in Encoding.UTF8.GetBytes(match.Value))
                    {
                        builder.Append(_byteEncoder[b]);
                    }
                    List<string> bpeTokens = Bpe(builder.ToString());
                    foreach (string token in bpeTokens)
                    {
                        tokenIds.Add(_encoder.TryGetValue(token, out int id) ? id : UnkTokenId);
                    }
                }
            }
        }
        return tokenIds;
    }

    public string Decode(List<int> tokenIds)
    {
        var builder = new StringBuilder();
        foreach (int id in tokenIds)
        {
            if (id == UnkTokenId) continue;
            builder.Append(_decoder.TryGetValue(id, out string token) ? token : "");
        }
        var byteBuffer = new List<byte>();
        foreach (char c in builder.ToString())
        {
            if (_byteDecoder.TryGetValue(c, out byte b))
            {
                byteBuffer.Add(b);
            }
        }
        return Encoding.UTF8.GetString(byteBuffer.ToArray());
    }

    private List<string> Bpe(string token)
    {
        if (_bpeCache.TryGetValue(token, out var cachedResult)) return cachedResult;
        if (token.Length <= 1)
        {
            var result = new List<string> { token };
            _bpeCache[token] = result;
            return result;
        }
        var word = token.Select(c => c.ToString()).ToList();
        while (true)
        {
            var pairs = GetPairs(word);
            if (pairs.Count == 0) break;
            var bestPair = pairs.OrderBy(p => _bpeRanks.GetValueOrDefault(p, int.MaxValue)).First();
            if (!_bpeRanks.ContainsKey(bestPair)) break;
            var newWord = new List<string>();
            int i = 0;
            while (i < word.Count)
            {
                if (i < word.Count - 1 && word[i] == bestPair.Item1 && word[i + 1] == bestPair.Item2)
                {
                    newWord.Add(bestPair.Item1 + bestPair.Item2);
                    i += 2;
                }
                else
                {
                    newWord.Add(word[i]);
                    i++;
                }
            }
            word = newWord;
            if (word.Count == 1) break;
        }
        _bpeCache[token] = word;
        return word;
    }
    
    private static HashSet<(string, string)> GetPairs(List<string> word)
    {
        var pairs = new HashSet<(string, string)>();
        if (word.Count < 2) return pairs;
        for (int i = 0; i < word.Count - 1; i++)
        {
            pairs.Add((word[i], word[i + 1]));
        }
        return pairs;
    }

    private static Dictionary<(string, string), int> LoadMergesFromString(string mergesContent)
    {
        var ranks = new Dictionary<(string, string), int>();
        var lines = mergesContent.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        int rank = 0;
        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
            var parts = line.Split(' ');
            if (parts.Length == 2) ranks[(parts[0], parts[1])] = rank++;
        }
        return ranks;
    }

    private static (Dictionary<byte, char>, Dictionary<char, byte>) BuildByteToUnicodeMap()
    {
        var byteToUnicode = new Dictionary<byte, char>();
        var unicodeToByte = new Dictionary<char, byte>();
        var visibleChars = new List<byte>();
        for (int i = '!'; i <= '~'; i++) visibleChars.Add((byte)i);
        for (int i = '¡'; i <= '¬'; i++) visibleChars.Add((byte)i);
        for (int i = '®'; i <= 'ÿ'; i++) visibleChars.Add((byte)i);
        var charSet = new HashSet<byte>(visibleChars);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            char mappedChar = !charSet.Contains((byte)b) ? (char)(256 + n++) : (char)b;
            byteToUnicode[(byte)b] = mappedChar;
            unicodeToByte[mappedChar] = (byte)b;
        }
        return (byteToUnicode, unicodeToByte);
    }
    
    private class TokenizerConfig
    {
        [JsonProperty("eos_token")]
        public string EosToken { get; set; }

        [JsonProperty("pad_token")]
        public string PadToken { get; set; }

        [JsonProperty("unk_token")]
        public string UnkToken { get; set; }

        [JsonProperty("added_tokens_decoder")]
        public Dictionary<string, AddedTokenDef> AddedTokensDecoder { get; set; }
    }

    private class AddedTokenDef
    {
        [JsonProperty("content")]
        public string Content { get; set; }
        
        [JsonProperty("special")]
        public bool Special { get; set; }
    }
}