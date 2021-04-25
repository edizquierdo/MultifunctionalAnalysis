// ***********************************************************
//  Methods for a VisualAgent
//
//  Matthew Setzler 4/19/17
//  Eduardo Izquierdo 4/13/19
// ************************************************************

#include "VisualAgent.h"

// Utility method for initializing and updating rays

void ResetRay(Ray &ray, double theta, double cx, double cy) {
  if (abs(theta) < 0.0000001) ray.m = INFINITY;  // special case, vertical ray
  else ray.m = 1/tan(theta);
  ray.b = cy - ray.m*cx;
  ray.length = MaxRayLength;

  // Set starting coordinates (i.e. on upper perimeter of agent body)
  if (ray.m == INFINITY) {
    ray.startX = cx;
    ray.startY = cy+BodySize/2;
    return;
  }
  ray.startX = cx + (BodySize/2) * sin(theta);
  ray.startY = cy + (BodySize/2) * cos(theta);
}


// *******
// Control
// *******

// Change x-position

void VisualAgent::SetPositionX(double newx) {
  cx = newx;
  ResetRays();
}

// Reset the state of the agent

void VisualAgent::Reset(double ix, double iy, int randomize) {
  cx = ix; cy = iy; vx = 0.0;
  if (randomize) NervousSystem.RandomizeCircuitState(-0.1, 0.1);
  else NervousSystem.RandomizeCircuitState(0.0, 0.0);
  ResetRays();
}

void VisualAgent::Reset(RandomState &rs, double ix, double iy, int randomize) {
  cx = ix; cy = iy; vx = 0;
  if (randomize) NervousSystem.RandomizeCircuitState(-0.1, 0.1, rs);
  else NervousSystem.RandomizeCircuitState(0.0, 0.0, rs);
  ResetRays();
}

void VisualAgent::ResetRays() {
  double theta = -VisualAngle/2;
  for (int i=1; i<=NumRays; i++) {
    ResetRay(Rays[i], theta, cx, cy);
    theta += VisualAngle/(NumRays-1);
  }
}

//
#ifdef GPM_FULL
void VisualAgent::SetController(TVector<double> &phen)
{
  int k = 1;
  // Weights of the neural network
  for (int i = 1; i <= NumNeurons; i++){
    for (int j = 1; j <= NumNeurons; j++){
      NervousSystem.SetConnectionWeight(i,j,phen(k));
      k++;
    }
  }
  // Biases
  for (int i = 1; i <= NumNeurons; i++){
    NervousSystem.SetNeuronBias(i,phen(k));
    k++;
  }
  // TimeConstants
  for (int i = 1; i <= NumNeurons; i++){
    NervousSystem.SetNeuronTimeConstant(i,phen(k));
    k++;
  }
  // Gains
  for (int i = 1; i <= NumNeurons; i++){
    NervousSystem.SetNeuronGain(i,phen(k));
    k++;
  }
}
#endif
#ifdef GPM_LAYER
void VisualAgent::SetController(TVector<double> &phen)
{
  int k = 1;
  // Weights from Sensor to Inter
  for (int i = 1; i <= NumRays; i++){
    for (int j = NumRays+1; j <= NumRays+NumInter; j++){
      NervousSystem.SetConnectionWeight(i,j,phen(k));
      k++;
    }
  }
  // Weights from Inter to Inter
  for (int i = NumRays+1; i <= NumRays+NumInter; i++){
    for (int j = NumRays+1; j <= NumRays+NumInter; j++){
      NervousSystem.SetConnectionWeight(i,j,phen(k));
      k++;
    }
  }
  // Weights from Inter to Motor
  for (int i = NumRays+1; i <= NumRays+NumInter; i++){
    for (int j = NumRays+NumInter+1; j <= NumRays+NumInter+NumMotor; j++){
      NervousSystem.SetConnectionWeight(i,j,phen(k));
      k++;
    }
  }
  // Biases: Interneurons + One for sensor and one for motor
  for (int i = 1; i <= NumRays; i++){
    NervousSystem.SetNeuronBias(i,phen(k));
  }
  k++;
  for (int i = NumRays+1; i <= NumRays+NumInter; i++){
    NervousSystem.SetNeuronBias(i,phen(k));
    k++;
  }
  for (int i = NumRays+NumInter+1; i <= NumRays+NumInter+NumMotor; i++){
    NervousSystem.SetNeuronBias(i,phen(k));
  }
  k++;
  // TimeConstants: Interneurons + One for sensor and one for motor
  for (int i = 1; i <= NumRays; i++){
    NervousSystem.SetNeuronTimeConstant(i,phen(k));
  }
  k++;
  for (int i = NumRays+1; i <= NumRays+NumInter; i++){
    NervousSystem.SetNeuronTimeConstant(i,phen(k));
    k++;
  }
  for (int i = NumRays+NumInter+1; i <= NumRays+NumInter+NumMotor; i++){
    NervousSystem.SetNeuronTimeConstant(i,phen(k));
  }
  k++;
  // Gains: Interneurons + One for sensor and one for motor
  for (int i = 1; i <= NumRays; i++){
    NervousSystem.SetNeuronGain(i,phen(k));
  }
  k++;
  for (int i = NumRays+1; i <= NumRays+NumInter; i++){
    NervousSystem.SetNeuronGain(i,phen(k));
    k++;
  }
  for (int i = NumRays+NumInter+1; i <= NumRays+NumInter+NumMotor; i++){
    NervousSystem.SetNeuronGain(i,phen(k));
  }
  k++;
}
#endif
#ifdef GPM_LAYER_SYM_ODD
void VisualAgent::SetController(TVector<double> &phen)
{
  int k = 1;
  // Weights from Sensor to Inter
  for (int i = 1; i <= H_NumRays - 1; i++){
    for (int j = NumRays+1; j <= NumRays+NumInter; j++){
      NervousSystem.SetConnectionWeight(i,j,phen(k));
      NervousSystem.SetConnectionWeight(NumRays-i+1,NumRays+NumInter-j+NumRays+1,phen(k));
      //cout << i << "," << j << " " << NumRays-i+1 << "," << NumRays+NumInter-j+NumRays+1 << endl;
      k++;
    }
  }
  for (int j = NumRays+1; j <= NumRays+H_NumInter; j++){
    NervousSystem.SetConnectionWeight(H_NumRays,j,phen(k));
    NervousSystem.SetConnectionWeight(H_NumRays,NumRays+NumInter-j+NumRays+1,phen(k));
    //cout << H_NumRays << "," << j << " " << H_NumRays << "," << NumRays+NumInter-j+NumRays+1 << endl;
    k++;
  }
  // Weights from Inter to Inter
  for (int i = NumRays+1; i <= NumRays+H_NumInter-1; i++){
    for (int j = NumRays+1; j <= NumRays+NumInter; j++){
      NervousSystem.SetConnectionWeight(i,j,phen(k));
      NervousSystem.SetConnectionWeight(NumRays+NumInter-i+NumRays+1,NumRays+NumInter-j+NumRays+1,phen(k));
      //cout << i << "," << j << " " << NumRays+NumInter-i+NumRays+1 << "," << NumRays+NumInter-j+NumRays+1 << endl;
      k++;
    }
  }
  for (int j = NumRays+1; j <= NumRays+H_NumInter; j++){
    NervousSystem.SetConnectionWeight(NumRays+H_NumInter,j,phen(k));
    NervousSystem.SetConnectionWeight(NumRays+H_NumInter,NumRays+NumInter-j+NumRays+1,phen(k));
    //cout << NumRays+H_NumInter << "," << j << " " << NumRays+H_NumInter << "," << NumRays+NumInter-j+NumRays+1 << endl;
    k++;
  }
  // Weights from Inter to Motor
  for (int i = NumRays+1; i <= NumRays+H_NumInter-1; i++){
    for (int j = NumRays+NumInter+1; j <= NumRays+NumInter+NumMotor; j++){
      NervousSystem.SetConnectionWeight(i,j,phen(k));
      NervousSystem.SetConnectionWeight(NumRays+NumInter-i+NumRays+1,NumRays+NumInter+NumMotor-j+NumRays+NumInter+1,phen(k));
      //cout << i << "," << j << " " << NumRays+NumInter-i+NumRays+1 << "," << NumRays+NumInter+NumMotor-j+NumRays+NumInter+1 << endl;
      k++;
    }
  }
  NervousSystem.SetConnectionWeight(NumRays+H_NumInter,NumRays+NumInter+NumMotor-1,phen(k));
  NervousSystem.SetConnectionWeight(NumRays+H_NumInter,NumRays+NumInter+NumMotor,phen(k));
  //cout << NumRays+H_NumInter << "," << NumRays+NumInter+NumMotor-1 << " " << NumRays+H_NumInter << "," << NumRays+NumInter+NumMotor << endl;
  k++;
  // Biases: Interneurons + One for sensor and one for motor
  for (int i = 1; i <= NumRays; i++){
    NervousSystem.SetNeuronBias(i,phen(k));
  }
  k++;
  for (int i = NumRays+1; i <= NumRays+H_NumInter; i++){
    NervousSystem.SetNeuronBias(i,phen(k));
    NervousSystem.SetNeuronBias(NumRays+NumInter-i+NumRays+1,phen(k));
    k++;
  }
  for (int i = NumRays+NumInter+1; i <= NumRays+NumInter+NumMotor; i++){
    NervousSystem.SetNeuronBias(i,phen(k));
  }
  k++;
  // TimeConstants: Interneurons + One for sensor and one for motor
  for (int i = 1; i <= NumRays; i++){
    NervousSystem.SetNeuronTimeConstant(i,phen(k));
  }
  k++;
  for (int i = NumRays+1; i <= NumRays+H_NumInter; i++){
    NervousSystem.SetNeuronTimeConstant(i,phen(k));
    NervousSystem.SetNeuronTimeConstant(NumRays+NumInter-i+NumRays+1,phen(k));
    k++;
  }
  for (int i = NumRays+NumInter+1; i <= NumRays+NumInter+NumMotor; i++){
    NervousSystem.SetNeuronTimeConstant(i,phen(k));
  }
  k++;
  // Gains: Interneurons + One for sensor and one for motor
  for (int i = 1; i <= NumRays; i++){
    NervousSystem.SetNeuronGain(i,phen(k));
  }
  k++;
  for (int i = NumRays+1; i <= NumRays+H_NumInter; i++){
    NervousSystem.SetNeuronGain(i,phen(k));
    NervousSystem.SetNeuronGain(NumRays+NumInter-i+NumRays+1,phen(k));
    k++;
  }
  for (int i = NumRays+NumInter+1; i <= NumRays+NumInter+NumMotor; i++){
    NervousSystem.SetNeuronGain(i,phen(k));
  }
  k++;
}
#endif
#ifdef GPM_LAYER_SYM_EVEN
void VisualAgent::SetController(TVector<double> &phen)
{
  int k = 1;
  // Weights from Sensor to Inter
  for (int i = 1; i <= H_NumRays; i++){
    for (int j = NumRays+1; j <= NumRays+NumInter; j++){
      NervousSystem.SetConnectionWeight(i,j,phen(k));
      NervousSystem.SetConnectionWeight(NumRays-i+1,NumRays+NumInter-j+NumRays+1,phen(k));
      //cout << i << "," << j << " " << NumRays-i+1 << "," << NumRays+NumInter-j+NumRays+1 << endl;
      k++;
    }
  }
  // Weights from Inter to Inter
  for (int i = NumRays+1; i <= NumRays+H_NumInter; i++){
    for (int j = NumRays+1; j <= NumRays+NumInter; j++){
      NervousSystem.SetConnectionWeight(i,j,phen(k));
      NervousSystem.SetConnectionWeight(NumRays+NumInter-i+NumRays+1,NumRays+NumInter-j+NumRays+1,phen(k));
      //cout << i << "," << j << " " << NumRays+NumInter-i+NumRays+1 << "," << NumRays+NumInter-j+NumRays+1 << endl;
      k++;
    }
  }
  // Weights from Inter to Motor
  for (int i = NumRays+1; i <= NumRays+H_NumInter; i++){
    for (int j = NumRays+NumInter+1; j <= NumRays+NumInter+NumMotor; j++){
      NervousSystem.SetConnectionWeight(i,j,phen(k));
      NervousSystem.SetConnectionWeight(NumRays+NumInter-i+NumRays+1,NumRays+NumInter+NumMotor-j+NumRays+NumInter+1,phen(k));
      //cout << i << "," << j << " " << NumRays+NumInter-i+NumRays+1 << "," << NumRays+NumInter+NumMotor-j+NumRays+NumInter+1 << endl;
      k++;
    }
  }
  // Biases: Interneurons + One for sensor and one for motor
  for (int i = 1; i <= NumRays; i++){
    NervousSystem.SetNeuronBias(i,phen(k));
  }
  k++;
  for (int i = NumRays+1; i <= NumRays+H_NumInter; i++){
    NervousSystem.SetNeuronBias(i,phen(k));
    NervousSystem.SetNeuronBias(NumRays+NumInter-i+NumRays+1,phen(k));
    k++;
  }
  for (int i = NumRays+NumInter+1; i <= NumRays+NumInter+NumMotor; i++){
    NervousSystem.SetNeuronBias(i,phen(k));
  }
  k++;
  // TimeConstants: Interneurons + One for sensor and one for motor
  for (int i = 1; i <= NumRays; i++){
    NervousSystem.SetNeuronTimeConstant(i,phen(k));
  }
  k++;
  for (int i = NumRays+1; i <= NumRays+H_NumInter; i++){
    NervousSystem.SetNeuronTimeConstant(i,phen(k));
    NervousSystem.SetNeuronTimeConstant(NumRays+NumInter-i+NumRays+1,phen(k));
    k++;
  }
  for (int i = NumRays+NumInter+1; i <= NumRays+NumInter+NumMotor; i++){
    NervousSystem.SetNeuronTimeConstant(i,phen(k));
  }
  k++;
  // TimeConstants: Interneurons + One for sensor and one for motor
  for (int i = 1; i <= NumRays; i++){
    NervousSystem.SetNeuronGain(i,phen(k));
  }
  k++;
  for (int i = NumRays+1; i <= NumRays+H_NumInter; i++){
    NervousSystem.SetNeuronGain(i,phen(k));
    NervousSystem.SetNeuronGain(NumRays+NumInter-i+NumRays+1,phen(k));
    k++;
  }
  for (int i = NumRays+NumInter+1; i <= NumRays+NumInter+NumMotor; i++){
    NervousSystem.SetNeuronGain(i,phen(k));
  }
  k++;
}
#endif

// Step the agent

void VisualAgent::Step(RandomState &rs, double StepSize, VisualObject &object) {
  // Update visual sensors and check inputs
  ResetRays();
  for (int i=1; i<=NumRays; i++) {
    object.RayIntersection(Rays[i]);
    ExternalInput[i] = InputGain*(MaxRayLength - Rays[i].length)/MaxRayLength;
    ExternalInput[i] += rs.GaussianRandom(0.0,SensorNoiseVar);
    NervousSystem.SetNeuronExternalInput(i, ExternalInput[i]);
    // NervousSystem.SetNeuronOutput(i,(MaxRayLength - Rays[i].length)/MaxRayLength);
  }

  // Step nervous system
  NervousSystem.EulerStep(StepSize);

  // Update agent state
  vx = VelGain*(NervousSystem.outputs[NumNeurons-1] - NervousSystem.outputs[NumNeurons]);
  cx = cx + StepSize*vx;
  cx += rs.GaussianRandom(0.0,MotorNoiseVar);
  if (cx < -EnvWidth/2) {
    cx = -EnvWidth/2;
  } else if (cx > EnvWidth/2) {
    cx = EnvWidth/2;
  }
}

void VisualAgent::Step2(RandomState &rs, double StepSize, VisualObject &object1, VisualObject &object2) {
  // Update visual sensors and check inputs
  ResetRays();
  for (int i=1; i<=NumRays; i++) {
    object1.RayIntersection(Rays[i]);
    object2.RayIntersection(Rays[i]);
    ExternalInput[i] = InputGain*(MaxRayLength - Rays[i].length)/MaxRayLength;
    ExternalInput[i] += rs.GaussianRandom(0.0,SensorNoiseVar);
    NervousSystem.SetNeuronExternalInput(i, ExternalInput[i]);
  }

  // Step nervous system
  NervousSystem.EulerStep(StepSize);

  // Update agent state
  vx = VelGain*(NervousSystem.outputs[NumNeurons-1] - NervousSystem.outputs[NumNeurons]);
  cx = cx + StepSize*vx;
  cx += rs.GaussianRandom(0.0,MotorNoiseVar);
  if (cx < -EnvWidth/2) {
    cx = -EnvWidth/2;
  } else if (cx > EnvWidth/2) {
    cx = EnvWidth/2;
  }
}
