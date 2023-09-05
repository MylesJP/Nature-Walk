# Nature Walk
An MMS service that allows the user to take a photo of a flower, send it to a phone number, and receive the classifiction of the flower in return. Here is an outline of how it works:
- Uses Twilio as an MMS service and hosts the receiving phone number
- Node server takes the image from Twilio and passes it so a Flask microservice being hosted locally on a Raspberry Pi.
- Flask microservice uses a custom EfficientNet_B2 image classificiation PyTorch model to classify the flower (hosted on HuggingFace at https://huggingface.co/spaces/MylesJP/Nature-Walk)
- Response from microservice is parsed and the result is sent back to the user as an SMS message.

## Example Images
<img src="https://github.com/MylesJP/Nature-Walk/assets/96148570/071d1940-a79a-4516-95d4-ebd1e1649a8d" width="300"/>
<img src="https://github.com/MylesJP/Nature-Walk/assets/96148570/18e402dc-e9ed-4634-9b50-faa7faa6c41c" width="300"/>

