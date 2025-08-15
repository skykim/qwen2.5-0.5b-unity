using UnityEngine;

public class MoveUpDown : MonoBehaviour
{
    public float moveHeight = 1f;
    public float moveSpeed = 2f;
    private Vector3 initialPosition;

    void Start()
    {
        initialPosition = transform.position;
    }

    void Update()
    {
        float newY = initialPosition.y + moveHeight * Mathf.Sin(Time.time * moveSpeed);
        transform.position = new Vector3(initialPosition.x, newY, initialPosition.z);
    }
}