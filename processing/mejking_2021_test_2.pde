// Mejking 2021 Festival
// Visual Identity
// test app
// by @fallenartist

import processing.video.*;
import processing.sound.*;
import gab.opencv.*;
import java.awt.Rectangle;

Capture cam;
OpenCV opencv;
boolean camReady = false; 
Rectangle[] faces;

color black = color(0, 0, 0);
color green = color(64, 128, 64);
color blue = color(0, 126, 192);
color mustard = color(187, 162, 24);
color white = color(255, 255, 255);

int num_images = 8;
PImage[] images = new PImage[num_images];
PImage face_img;
PGraphics pg;

IntList ii = new IntList();
int[] iish;

int seed = int(random(1000000));
int[] sizes = { 20, 40, 80 };
int min_step = min(sizes);
int max_step = max(sizes);
int step;

int fx = 0, fy = 0, fw = 0, fh = 0, fi;
float cl, d0, d1, dz, dv, rvx, rvy, ratio;
float[] rr;

PRNG prng = new PRNG();

void setup() {
  size(600, 800);
  surface.setResizable(false); 
  frameRate(25);
  noStroke();
  
  String[] cameras = Capture.list();
  println("Available cameras:");
  printArray(cameras);
  
  pg = createGraphics(width, height);

  randomSeed(seed);
  
  ratio = float(width) / float(height);
  d0 = random(100, 200);
  d1 = random(25, 75);
  dz = random(0, 100);
  dv = random(0, 0.1);
  
  rr = new float[(width / max_step * height / max_step) / int(random(8, 13)) * 2];
  for (int i = 0; i < rr.length; i+=2) {
    rr[i] = random(width / max_step);
    rr[i+1] = random(height / max_step);
  }

  textFont(createFont("IBMPlexMono-Medium.ttf", 50));
  textAlign(LEFT, TOP);
  
  for (int i = 0; i < num_images; i++) ii.append(i);
  ii.shuffle();
  iish = ii.array();

  for (int i = 0; i < num_images; i++) {
    String image_name = "data/F" + nf(i, 2) + ".png";
    images[i] = loadImage(image_name);
  }
  
  cam = new Capture(this, "pipeline:avfvideosrc");
  //cam = new Capture(this, 600, 800, cameras[0], 30);
  opencv = new OpenCV(this, cam.width, cam.height);
  //opencv = new OpenCV(this, cam);
  opencv.loadCascade(OpenCV.CASCADE_FRONTALFACE);
  cam.start();

}

void draw() {
  
  background(blue);
  
  if (frameCount == 1) {           // GL context is current now
    cam = new Capture(this, "pipeline:autovideosrc");  // ← safest on macOS
    cam.start();
    return;                        // give it a frame to settle
  }
  
  if (!camReady && cam.available()) {
    cam.read();                    // now cam.width / height are valid
    opencv = new OpenCV(this, cam.width, cam.height);
    opencv.loadCascade(OpenCV.CASCADE_FRONTALFACE);
    camReady = true;               // OK, full pipeline is live
  }
  
  if (!camReady) {
    background(0);
    fill(255); text("initialising camera …", 40, 40);
    return;
  }
  
  if (cam.available()) {
    cam.read();
  }
  
  // detect faces
  opencv.loadImage(cam);
  faces = opencv.detect();
  fw = 0;
  fi = -1;
  for (int i = 0; i < faces.length; i++) {
    if (faces[i].width > fw) {
      fx = faces[i].x;
      fy = faces[i].y;
      fw = faces[i].width;
      fh = faces[i].height;
      fi = i;
    }
  }
  
  if (fi > -1) { // face detected
    pg.beginDraw(); // added
    pg.image(cam, 0, 0);
    pg.copy(pg, int(fx - (fw * ratio - fw) / 2), fy, int(fw * ratio), fh, 0, 0, width, height);
    pg.filter(GRAY);
    pg.endDraw(); // added
  }

  // map
  for (int p = 0; p < sizes.length; p++) {
    
    step = sizes[p];

    for (int j = 0; j < height; j += step) {
      for (int i = 0; i < width; i += step) {
        
        float n0 = prng.perlin(i / (d0 + dz), j / (d0 + dz));
        float n1 = prng.perlin(i / (d1 + dz), j / (d1 + dz));
        float n = 1 - (n0 * 0.75 + n1 * 0.25);
        
        if (fi > -1) { // face
          
          //float n = brightness(pg.get(i, j)) / 256;
          //n = 0.5;
          println("face");
          
        }
        
        int k = int(n * num_images);
        
        if (p == 0) { // draw all

          fill(lerpColor(blue, white, iish[k]/10.0));
          rect(i, j, step, step);
          image(images[iish[k]], i, j, step, step);
        
        } else if (p == 1 && iish[k] < 4) { // draw some of the shades
          
          fill(lerpColor(blue, white, iish[k]/10.0));
          rect(i, j, step, step);
          image(images[iish[k]], i, j, step, step);
          
        } else if (p == 2) {
          
          for (int r = 0; r < rr.length; r += 2) {
            if (i == int(rr[r]) * step && j == int(rr[r+1]) * step) {
              fill(lerpColor(blue, white, iish[k]/10.0));
              rect(i, j, step, step);
              image(images[iish[k]], i, j, step, step);
            }
          }
          
        }

      }
    }
    
  }

  // typography

  fill(255);
  rect(0, min_step * 2, min_step * 6, min_step);
  fill(0);
  textSize(min_step * 0.7);
  text("mejking 2021", min_step / 2, min_step * 2);

  fill(255);
  rect(0, height - min_step, width, min_step);
  fill(0);
  text("#" + seed, min_step / 2, height - min_step + min_step / 8);

  // move stuff
  dz += dv;
  
  //noLoop();
  
}

void keyPressed() {
  
  seed = int(random(1000000));
  randomSeed(seed);
  d0 = random(100, 200);
  d1 = random(25, 75);
  dz = random(0, 100);
  dv = random(0, 0.1);
  
  ii.shuffle();
  iish = ii.array();
  
  rr = new float[(width / max_step * height / max_step) / int(random(8, 13)) * 2];
  for (int i = 0; i < rr.length; i+=2) {
    rr[i] = random(width / max_step);
    rr[i+1] = random(height / max_step);
  }
  
  //loop();
  
}
