const debug = false;

let videoGrabber;
let poseNet;
let pose;
let bodySkeleton;
let neuralNetwork;

let QRCode;
let logo;
let soundtrack;

const JUMPINGJACK_UP = "u";
const JUMPINGJACK_DOWN = "d";
const KEY_MODEL_TRAIN = "t";
const KEY_SAVE_DATA = "s";

let state;
const STATES = {
  WAITING: "waiting",
  COLLECTING: "collecting",
  TRAINING: "training",
  READY: "ready",
  WIN: "win",
  LOSE: "lose"
}


let targetLabel;
let resultLabel;
let playerSequence = "ud";
let gameScore = 0;
let scoreToWin = 8;

//timer to beat the game. Might started delayed due to loading time
let timer = 20;

const INSTRUCTIONS = {
  GO_UP: "MOVE ARMS UP!",
  GO_DOWN: "MOVE ARMS DOWN!"
};

let instructionMsg = INSTRUCTIONS.GO_UP;



//load assets before main sketch
function preload() {

  QRCode = loadImage("assets/qr-code.png");
  logo = loadImage("assets/logo.svg");
  soundtrack = loadSound("assets/soundtrack.mp3");

}

//initialize and setup main objects 
function setup() {

  createCanvas(windowWidth, windowHeight);
  videoGrabber = createCapture(VIDEO);
  videoGrabber.hide();

  poseNet = ml5.poseNet(videoGrabber, modelLoaded);
  poseNet.on('pose', gotPoses);

  //configuration options for neural network.
  //inputs are the x,y of the 17 body parts, outputs are the 2 possible poses
  //see more here: https://learn.ml5js.org/docs/#/reference/neural-network
  let config = {
    inputs: 34,
    outputs: 2,
    task: "classification",
    debug: true
  };

  neuralNetwork = ml5.neuralNetwork(config);

  let modelFiles = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin'
  };

  //loads myo own trained data from assets folder
  neuralNetwork.load(modelFiles, posesModelLoaded);

  //timer every 1 second for gameplay
  setInterval(decreaseTimer, 1000);
  soundtrack.loop();

}



function draw() {
  background(10);
  image(logo, 30, 10);

  //if game is lost
  if (state == STATES.LOSE) {
    push();
    textAlign(CENTER, CENTER);
    textSize(40);
    fill(255);
    text("You Lost..", width / 2, height / 2);
    pop();
    return;
  }
  //if game is won
  if (state == STATES.WIN) {
    push();
    fill(255);
    textAlign(CENTER, CENTER);
    textSize(30);
    text("You Win!", width / 2, 20);
    imageMode(CENTER);
    image(QRCode, width / 2, height / 2);
    text("Scan to unlock your price!", width / 2, height - 60);
    pop();

    return;
  }




  //if game is active
  if (state == STATES.READY) {
    push();
    //video setup, mirrored and centered
    translate(width, 0);
    scale(-1, 1);
    let marginX = (windowWidth - videoGrabber.width);
    let marginY = (windowHeight - videoGrabber.height);
    translate(marginX / 2, marginY / 2);

    image(videoGrabber, 0, 0, videoGrabber.width, videoGrabber.height);


    //if there is a pose found...
    if (pose) {

      //global flag. assign to true to see joints in body
      if (debug) {
        // draw a ball in every join
        for (let i = 0; i < pose.keypoints.length; i++) {
          let x = pose.keypoints[i].position.x;
          let y = pose.keypoints[i].position.y;
          fill(0, 250, 0);
          ellipse(x, y, 20, 20);
        }

        //draw the skeleton between the points.
        for (let i = 0; i < bodySkeleton.length; i++) {
          let partA = bodySkeleton[i][0];
          let partB = bodySkeleton[i][1];
          strokeWeight(4);
          stroke(255);
          line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
        }

      }

      pop();


      //Misc UI messages
      textSize(40);
      textAlign(LEFT);
      fill(255);
      text(`Score: ${gameScore}/${scoreToWin}`, 20, height - 20);
      text(`Time Left: ${timer}`, 20, height - 60);

      textAlign(CENTER, CENTER);
      text(`${instructionMsg}`, width / 2, 40);

      textAlign(CENTER);
      textSize(20);
      text("Give me some jumping jacks!", width / 2, height - 30);
    }

  }


}

//Event called when poseNet finds a pose
function gotPoses(poses) {

  //Assume there is only one pose/person. Get that pose and skeleton and store in global variable
  if (poses.length > 0) {
    pose = poses[0].pose;
    bodySkeleton = poses[0].skeleton;

    if (state == STATES.READY) {

      //turn poseNet positions from array to what the neural net expects
      let inputs = flattenPositionArray(pose);

      //classify data
      neuralNetwork.classify(inputs, classificationResult);
    }
  }
}



//Callback for when the neural network classifies the data
function classificationResult(err, results) {

  if (!err) {
    //process the result
    keepScore(results);
  } else {
    console.log(err);
  }

}


//determine game effects of the resulting data
function keepScore(results) {


  //if the pose confidence is lower than 70% don't take is as an input
  if (results[0].confidence < 0.7) {
    console.log(`low confidence ${results[0].confidence}`);
    return;
  }

  //get the current pose
  let resultPose = results[0].label;

  //get the last pose by the player
  let lastPose = playerSequence[playerSequence.length - 1];


  //if the new pose is different than the previous one, trigger a change in pose
  if (lastPose != resultPose) {

    //get the last two poses by the player
    let lastTwoPoses = playerSequence[playerSequence.length - 2] +
      playerSequence[playerSequence.length - 1];

    //if the player is coming from a down to up motion..
    if (lastTwoPoses == "ud") {
      //if the new resulting pose is an up, set instructions to go down
      if (resultPose == "u") {
        instructionMsg = INSTRUCTIONS.GO_DOWN;
      }
    }

    //if the player is coming from an up to down motion..
    if (lastTwoPoses == "du") {
      //if the new resulting pose is a down, set instruction to go up and increase the score
      if (resultPose == "d") {
        instructionMsg = INSTRUCTIONS.GO_UP;
        gameScore++;
      }
    }

    //concatenate the last move to the sequence
    playerSequence += resultPose;
    // console.log(`added new pose to sequence-->\n ${playerSequence}`);
  }
}



//callback called every second by setInterval in setup. Keeps game play timing
function decreaseTimer() {

  timer--;
  //if time is up..
  if (timer == 0) {

    //check if game was beaten
    if (gameScore < scoreToWin) {
      //set state variable to lost. will affect draw loop
      state = STATES.LOSE;
      //wait 5 seconds and restart game
      setTimeout(restartGame, 5000);
    }
    else {
      //set state variable to won. will affect draw loop
      state = STATES.WIN;
      //wait 15 seconds so users can scan QR and restart game
      setTimeout(restartGame, 15000);
    }

  }

}





//take pose data and flatten x&y from object to an array that is fed to neuralNetwork
function flattenPositionArray(pose) {

  let positions = [];
  for (let i = 0; i < pose.keypoints.length; i++) {
    positions.push(pose.keypoints[i].position.x);
    positions.push(pose.keypoints[i].position.y);
  }

  return positions;
}


//Notify of loaded model for debugging
function modelLoaded() {
  console.log("ML5 model loaded. Ready to detect body");
}

//My data has been loaded. Set state to ready
function posesModelLoaded() {
  console.log("Poses data DA loaded");
  state = STATES.READY;
}

//restart game play variables
function restartGame() {
  state = STATES.READY;
  timer = 20;
  gameScore = 0;
}



