using UnityEngine;
using Unity.InferenceEngine;
using FF = Unity.InferenceEngine.Functional;
using System.IO;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.UI;
using System.Diagnostics;
using Debug = UnityEngine.Debug;
using System.Threading.Tasks;

public class ModelQwen : MonoBehaviour
{
    private Qwen2Tokenizer _tokenizer;
    private Worker _engine;
    private Worker _decodeEngine;

    private const BackendType BACKEND = BackendType.CPU;
    private const int MAX_LAYERS = 24;
    private const int NUM_KEY_VALUE_HEADS = 2;
    private const int HEAD_DIM = 64;
    private const int TOKEN_IM_END = 151645;

    private List<int> _tokens = new();
    private List<int> _outputTokens = new();

    private int _maxNewTokens = 100;
    private List<int> _eosTokens;

    [SerializeField]
    Text textLabel;
    string systemMessage = "You are a helpful assistant. Answer concisely and clearly in up to three sentences.";

    void Start()
    {
        string modelFilePath = Path.Combine(Application.streamingAssetsPath, "qwen2.5-0.5b_uint8.sentis");
        string vocabFilePath = Path.Combine(Application.streamingAssetsPath, "vocab.json");
        string mergeFilePath = Path.Combine(Application.streamingAssetsPath, "merges.txt");
        string configFilePath = Path.Combine(Application.streamingAssetsPath, "tokenizer_config.json");

        string vocabContent = File.ReadAllText(vocabFilePath);
        string mergesContent = File.ReadAllText(mergeFilePath);
        string configContent = File.ReadAllText(configFilePath);

        _tokenizer = new Qwen2Tokenizer(vocabContent, mergesContent, configContent);
        _eosTokens = new List<int> { 151645, 151643 };

        Model baseModel = ModelLoader.Load(modelFilePath);

        var vocab_size = 151936;
        FunctionalGraph graph = new FunctionalGraph();
        FunctionalTensor logitsInput = graph.AddInput<float>(new DynamicTensorShape(1, -1, vocab_size));
        FunctionalTensor argMax = FF.ArgMax(logitsInput, 2, false);
        Model greedyModel = graph.Compile(argMax);

        _engine = new Worker(baseModel, BACKEND);
        _decodeEngine = new Worker(greedyModel, BACKEND);

        Warmup();
    }

    private void Warmup()
    {
        Debug.Log("Warming up the model...");
        var stopwatch = new Stopwatch();
        stopwatch.Start();

        using (var dummyInput = new Tensor<int>(new TensorShape(1, 1), new[] { 1 }))
        using (var dummyAttentionMask = new Tensor<int>(new TensorShape(1, 1), new[] { 1 }))
        using (var dummyPositionIds = new Tensor<int>(new TensorShape(1, 1), new[] { 0 }))
        {
            _engine.SetInput("input_ids", dummyInput);
            _engine.SetInput("attention_mask", dummyAttentionMask);
            _engine.SetInput("position_ids", dummyPositionIds);

            var emptyPastShape = new TensorShape(1, NUM_KEY_VALUE_HEADS, 0, HEAD_DIM);
            using (var emptyPastTensor = new Tensor<float>(emptyPastShape))
            {
                for (int i = 0; i < MAX_LAYERS; i++)
                {
                    _engine.SetInput($"past_key_values.{i}.key", emptyPastTensor);
                    _engine.SetInput($"past_key_values.{i}.value", emptyPastTensor);
                }
            }

            _engine.Schedule();
            using var dummyLogits = _engine.PeekOutput("logits").ReadbackAndClone() as Tensor<float>;

            _decodeEngine.SetInput(0, dummyLogits);
            _decodeEngine.Schedule();
        }

        stopwatch.Stop();
        Debug.Log($"Warmup complete in {stopwatch.ElapsedMilliseconds} ms.");
    }

    public async void Generate(InputField inputPrompt)
    {
        string finalPrompt = $"<|im_start|>system\n{systemMessage}<|im_end|>\n<|im_start|>user\n{inputPrompt.text}<|im_end|>\n<|im_start|>assistant\n";
        Debug.Log("Prompt: " + finalPrompt);

        textLabel.text = "";

        _tokens.Clear();
        _tokens.AddRange(_tokenizer.Encode(finalPrompt));
        _outputTokens.Clear();

        var stopwatch = new Stopwatch();
        stopwatch.Start();

        int step = 0;
        int initialTokenCount = _tokens.Count;
        int[] initialTokens = _tokens.ToArray();
        int prefillSequenceLength = initialTokens.Length;

        using var inputTensor = new Tensor<int>(new TensorShape(1, prefillSequenceLength), initialTokens);
        using var attentionMaskTensor = new Tensor<int>(new TensorShape(1, prefillSequenceLength), Enumerable.Repeat(1, prefillSequenceLength).ToArray());
        using var positionIdsTensor = new Tensor<int>(new TensorShape(1, prefillSequenceLength), Enumerable.Range(0, prefillSequenceLength).ToArray());

        _engine.SetInput("input_ids", inputTensor);
        _engine.SetInput("attention_mask", attentionMaskTensor);
        _engine.SetInput("position_ids", positionIdsTensor);

        var emptyPastShape = new TensorShape(1, NUM_KEY_VALUE_HEADS, 0, HEAD_DIM);
        Tensor<float>[] pastKeys = new Tensor<float>[MAX_LAYERS];
        Tensor<float>[] pastValues = new Tensor<float>[MAX_LAYERS];

        for (int i = 0; i < MAX_LAYERS; i++)
        {
            pastKeys[i] = new Tensor<float>(emptyPastShape);
            pastValues[i] = new Tensor<float>(emptyPastShape);

            _engine.SetInput($"past_key_values.{i}.key", pastKeys[i]);
            _engine.SetInput($"past_key_values.{i}.value", pastValues[i]);
        }

        _engine.Schedule();

        using var outputLogits = _engine.PeekOutput("logits") as Tensor<float>;
        int nextToken = await ProcessLogits(outputLogits, prefillSequenceLength - 1);

        if (nextToken != TOKEN_IM_END)
        {
            _tokens.Add(nextToken);
            _outputTokens.Add(nextToken);
        }

        textLabel.text = _tokenizer.Decode(_outputTokens);

        step = 1;
        while (step < _maxNewTokens && !_eosTokens.Contains(nextToken))
        {
            for (int i = 0; i < MAX_LAYERS; i++)
            {
                pastKeys[i] = await _engine.PeekOutput($"present.{i}.key").ReadbackAndCloneAsync() as Tensor<float>;
                pastValues[i] = await _engine.PeekOutput($"present.{i}.value").ReadbackAndCloneAsync() as Tensor<float>;

                _engine.SetInput($"past_key_values.{i}.key", pastKeys[i]);
                _engine.SetInput($"past_key_values.{i}.value", pastValues[i]);
            }

            int currentSequenceLength = initialTokenCount + step;

            using var newInputTensor = new Tensor<int>(new TensorShape(1, 1), new[] { nextToken });
            using var newPositionIdsTensor = new Tensor<int>(new TensorShape(1, 1), new[] { currentSequenceLength - 1 });
            using var newAttentionMaskTensor = new Tensor<int>(new TensorShape(1, currentSequenceLength), Enumerable.Repeat(1, currentSequenceLength).ToArray());

            _engine.SetInput("input_ids", newInputTensor);
            _engine.SetInput("attention_mask", newAttentionMaskTensor);
            _engine.SetInput("position_ids", newPositionIdsTensor);

            _engine.Schedule();

            using var newOutputLogits = _engine.PeekOutput("logits") as Tensor<float>;
            nextToken = await ProcessLogits(newOutputLogits, 0);

            if (nextToken != TOKEN_IM_END)
            {
                _tokens.Add(nextToken);
                _outputTokens.Add(nextToken);
            }

            textLabel.text = _tokenizer.Decode(_outputTokens);
            //Debug.Log($"Step {step}: Generated token: {nextToken} - {_tokenizer.Decode(new List<int> { nextToken })}");

            step++;
        }

        for (int i = 0; i < MAX_LAYERS; i++)
        {
            pastKeys[i]?.Dispose();
            pastValues[i]?.Dispose();
        }

        string generatedText = _tokenizer.Decode(_outputTokens);
        Debug.Log($"Final sequence: {generatedText}");

        stopwatch.Stop();
        Debug.Log($"<color=cyan><b>Total Generation Time: {stopwatch.ElapsedMilliseconds} ms</b></color>");
    }

    private async Task<int> ProcessLogits(Tensor<float> logits, int sequenceIndex)
    {
        _decodeEngine.SetInput(0, logits);
        _decodeEngine.Schedule();
        using var argMaxTensor = await _decodeEngine.PeekOutput().ReadbackAndCloneAsync() as Tensor<int>;
        int nextToken = argMaxTensor.DownloadToArray()[sequenceIndex];
        //string tokenStr = _tokenizer.Decode(new List<int> { nextToken });
        return nextToken;
    }

    private void OnDestroy()
    {
        _engine?.Dispose();
        _decodeEngine?.Dispose();
    }
}