// ***********************************************************
//  A class for visual agents
//
//  Matthew Setzler 4/19/17
//  Eduardo Izquierdo 4/25/21
// ************************************************************

#include "CTRNN.h"
#include "VisualObject.h"

//#define GPM_FULL
//#define GPM_LAYER
#define GPM_LAYER_SYM_ODD
//#define GPM_LAYER_SYM_EVEN

// Global constants
const double Pi = 3.1415926535897;
const double BodySize = 30.0;  //**20.0; // diameter of agent
const double EnvWidth = 400.0;
const double MaxRayLength = 220.0;
const double InputGain = 1.0;  // Maximum input into the sensory neurons (from 0 to 10)
const double VisualAngle = Pi/6; //**Pi/6;
const double VelGain = 8.0; //**5.0 // sum of forces "constant of proportionality"
// const double SensorNoiseVar = 0.0; //0.1; //0.0; //0.1;
// const double MotorNoiseVar = 0.0; //0.01; //0.0; //0.01;

// The VisualAgent class declaration
class VisualAgent {
	public:
		// The constructor
		VisualAgent(double ix = 0.0, double iy = 0.0, int NumRays_ = 7, int NumInterNeurons_ = 5, int NumMotors_ = 2) {
			NumRays = NumRays_;
			Rays.SetBounds(1, NumRays);
			ExternalInput.SetBounds(1,NumRays);
			ExternalInput.FillContents(0.0);
			Reset(ix,iy);
			NumInter = NumInterNeurons_;
			NumMotor = NumMotors_;
			SensorWeight.SetBounds(1,NumRays,1,NumInter);
			SensorWeight.FillContents(0.0);
			H_NumRays = (int) ceil((float)NumRays_/2);
			H_NumInter = (int) ceil((float)NumInterNeurons_/2);
			H_NumMotor = (int) ceil((float)NumMotors_/2);
			NumNeurons = NumInterNeurons_ + NumMotors_;
			NervousSystem.SetCircuitSize(NumNeurons);
		};

		// The destructor
		~VisualAgent() {}

		// Accessors
		double PositionX() {return cx;};
		void SetPositionX(double newx);
		double PositionY() {return cy;};
		double VelocityX() {return vx;};

		void SetController(TVector<double> &phen);

		// Control
		void Reset(double ix, double iy, int randomize = 0);
    void Reset(RandomState &rs, double ix, double iy, int randomize);

		void Step(RandomState &rs, double StepSize, VisualObject &object);
		void StepNodeLesion(RandomState &rs, double StepSize, VisualObject &object, int lj, double outputj);
		void StepSensorEdgeLesion(RandomState &rs, double StepSize, VisualObject &object, int lj, int li, double outputj);
		void StepInterEdgeLesion(RandomState &rs, double StepSize, VisualObject &object, int from, int to, double outputj);
		void StepInterTwoWayEdgeLesion(RandomState &rs, double StepSize, VisualObject &object, int from, int to, double outputFrom, double outputTo);

		void Step2(RandomState &rs, double StepSize, VisualObject &object1, VisualObject &object2);
		void Step2NodeLesion(RandomState &rs, double StepSize, VisualObject &object1, VisualObject &object2, int lj, double outputj);
		void Step2SensorEdgeLesion(RandomState &rs, double StepSize, VisualObject &object1, VisualObject &object2, int lj, int li, double outputj);
		void Step2InterEdgeLesion(RandomState &rs, double StepSize, VisualObject &object1, VisualObject &object2, int lj, int li, double outputj);
		void Step2InterTwoWayEdgeLesion(RandomState &rs, double StepSize, VisualObject &object1, VisualObject &object2, int from, int to, double outputFrom, double outputTo);

		CTRNN NervousSystem;
		TVector<double> ExternalInput;
		TMatrix<double> SensorWeight;

	private:
		void ResetRays();

		int NumRays;
		int NumNeurons;
		int NumMotor;
		int NumInter;
		int H_NumRays;
		int H_NumInter;
		int H_NumMotor;
		double cx, cy, vx;
		TVector<Ray> Rays;
};
