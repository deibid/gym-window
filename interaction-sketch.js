let videoGrabber;
let poseNet;
let pose;
let bodySkeleton;
let neuralNetwork;

const JUMPINGJACK_UP = "u";
const JUMPINGJACK_DOWN = "d";
const KEY_MODEL_TRAIN = "t";
const KEY_SAVE_DATA = "s";

let state;
const STATES = {
  WAITING: "waiting",
  COLLECTING: "collecting",
  TRAINING: "training",
  READY: "ready"
}

let targetLabel;
let resultLabel;


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

  neuralNetwork.load(modelFiles, posesModelLoaded);

}



//control training process with keyboard and timers. press the key to train model
// for pose with a little delay 
function keyPressed() {

  if (key == KEY_MODEL_TRAIN) {
    //train data
    neuralNetwork.normalizeData();
    neuralNetwork.train({ epochs: 50 }, trainingFinished);

  } else if (key == JUMPINGJACK_DOWN || key == JUMPINGJACK_UP) {

    targetLabel = key;

    //two-callback system to toggle state variable from collecting to waiting
    setTimeout(() => {
      console.log(`Collecting data for ${key} pose ...`);
      state = STATES.COLLECTING;
      setTimeout(() => {
        console.log('Finished collecting.');
        state = STATES.WAITING;
      }, 3500);
    }, 2000);

  }

}

function draw() {
  background(255);
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

    //draw a ball in every join
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0, 250, 0);
      ellipse(x, y, 20, 20);
    }

    for (let i = 0; i < bodySkeleton.length; i++) {
      let partA = bodySkeleton[i][0];
      let partB = bodySkeleton[i][1];
      strokeWeight(4);
      stroke(255);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }

  }

  pop();

  if (state == STATES.COLLECTING) {
    textSize(60);
    textAlign(CENTER, CENTER);
    text(`${state} for ${targetLabel}`, width / 2, height - 20);
  }

  if (state == STATES.READY) {
    textSize(60);
    textAlign(CENTER, CENTER);
    text(`${resultLabel}`, width / 2, height - 20);
  }
}




function gotPoses(poses) {

  console.log("got pose");
  //Assume there is only one pose/person. Get that pose and skeleton and store in global variable
  if (poses.length > 0) {
    pose = poses[0].pose;
    bodySkeleton = poses[0].skeleton;
    //collect data for NN training
    if (state == STATES.COLLECTING) {
      let inputs = flattenPositionArray(pose);
      let target = [targetLabel];
      neuralNetwork.addData(inputs, target);
    }

    if (state == STATES.READY) {
      let inputs = flattenPositionArray(pose);
      neuralNetwork.classify(inputs, classificationResult);
    }

  }
}


function classificationResult(err, results) {

  console.log("Classfication ready");
  if (!err) {
    console.log(results);
    console.log(results[0].label);
    console.log(results[0].confidence);
    if (results[0].confidence > 0.7) {
      resultLabel = results[0].label;
    } else {
      resultLabel = "";
    }

  } else {
    console.log(err);
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

function posesModelLoaded() {
  console.log("Poses data DA loaded");
  state = STATES.READY;
}

function trainingFinished() {
  console.log('model trained');
  neuralNetwork.save();
}

