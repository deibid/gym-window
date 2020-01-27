// let promoScreen;
// let gameScreen;
// let qrScreen;


let videoGrabber;
let poseNet;
let pose;
let bodySkeleton;




function setup() {

  createCanvas(windowWidth, windowHeight);
  videoGrabber = createCapture(VIDEO);
  videoGrabber.hide();

  poseNet = ml5.poseNet(videoGrabber, modelLoaded);
  poseNet.on('pose', gotPoses);


  // promoScreen = document.getElementById("video-container");
  // gameScreen = document.getElementById("game-container");
  // qrScreen = document.getElementById("qr-container");

  // gameScreen.hide();
  // qrScreen.hide();


}

function draw() {

  translate(videoGrabber.width, 0);
  scale(-1, 1);

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



}


//Notify of loaded model for debugging
function modelLoaded() {
  console.log("ML5 model loaded. Ready to detect body");
}

function gotPoses(poses) {
  // console.log(poses);

  //Assume there is only one pose/person. Get that pose and skeleton and store in global variable
  if (poses.length > 0) {
    pose = poses[0].pose;
    bodySkeleton = poses[0].skeleton;
  }
}


function buttonPressed(number) {
  console.log(number);
  switch (number) {
    case 1:
      // let document.findElementById();
      break;
    case 2:
      break;
    case 3:
      break;
  }
}