const express = require("express");
const bodyParser = require("body-parser");
const axios = require("axios");
const twilio = require("twilio");
const fs = require("fs");
const FormData = require("form-data");
require("dotenv").config();

const app = express();
app.use(bodyParser.urlencoded({ extended: false }));

const client = new twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);

app.post("/sms", async (req, res) => {
  const twiml = new twilio.twiml.MessagingResponse();

  // Extract the image URL from the incoming MMS
  const imageUrl = req.body.MediaUrl0; // Assuming the image is the first media in the MMS

  try {
    // Call Flask API to get the prediction
    const imageBuffer = fs.readFileSync(imageUrl);

    const form = new FormData();
    form.append("image", imageBuffer, {
      filename: "image.jpg",
      contentType: "image/jpeg",
    });

    // Make POST request with FormData
    const response = await axios.post("http://192.168.0.79:5000/predict", form, {
      headers: {
        ...form.getHeaders(),
      },
    });

    // Send an SMS with the prediction
    client.messages.create({
      body: response.data,
      from: process.env.TWILIO_PHONE_NUMBER,
      to: req.body.From, // Send the response back to the original sender
    });
  } catch (error) {
    console.error("Error:", error);
    twiml.message("Sorry, something went wrong. Please try again.");
  }

  res.writeHead(200, { "Content-Type": "text/xml" });
  res.end(twiml.toString());
});

app.listen(3000, () => {
  console.log("Listening on port 3000...");
});
