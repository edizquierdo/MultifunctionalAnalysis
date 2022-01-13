
#include <iostream>
#include <iomanip>  // cout precision
#include "TSearch.h"
#include "random.h"
#include "VisualAgent.h"

// #define EVOLVE
// #define PRINTTOFILE

using namespace std;

// EA constants
const int	POPSIZE = 80;
const int GENS = 1000;

// Global constants
const int NUMRAYS = 15;
const int NUMINTER = 7;
const int NUMMOTOR = 2;
const int CIRCUITSIZE = NUMMOTOR + NUMINTER;
const int H_NUMRAYS = 8;
const int H_NUMINTER = 4;
const int H_NUMMOTOR = 1;

const double StepSize = 0.01;
const double MAXDISTANCE = 75.0;

const double PI = 3.14159265359;

// TRIAL VARIABLES
// Velocity
const double MINVEL = -3.0;
const double MAXVEL = -3.0;
const double VELSTEP = 0.5;
// Size
const double MINSIZE = 20.0; //20.0; // Diameter of agent is 30
const double MAXSIZE = 40.0; //40.0;
const double SIZESTEP = 1.0; //0.5;
// Pos
const double MINPOS = 1; //5;
const double MAXPOS = 5; //6; //10;
const double POSSTEP = 1; //5;
// Start vertical position of objects
const double STARTHEIGHT = 240; //275
const double OBJECTHEIGHT = 30;

const double REPS = 1;

// Circuit's genotype-phenotype mapping
const double WEIGHTMAX = 10.0;
const double BIASMAX = 10.0;
const double TAUMIN = 1.0;
const double TAUMAX = 2.0;
const double GAINMIN = 1.0;
const double GAINMAX = 5.0;

// Genotype size
#ifdef GPM_FULL
int	VectSize = (CIRCUITSIZE * CIRCUITSIZE) + (3 * CIRCUITSIZE);
#endif
#ifdef GPM_LAYER // Same one bias, timeconstant, and gain for all sensory neurons; Same one b, t, g, for the two motoneurons
int	VectSize = (NUMRAYS*NUMINTER + NUMINTER*NUMINTER + NUMINTER*NUMMOTOR) + 3*(NUMINTER+2);
#endif
#ifdef GPM_LAYER_SYM_ODD	// For odd number of interneurons
int	VectSize = ((H_NUMRAYS - 1)*NUMINTER + H_NUMINTER + (H_NUMINTER-1)*NUMINTER + H_NUMINTER + (H_NUMINTER-1)*NUMMOTOR + H_NUMMOTOR) + 3*(H_NUMINTER+1);
#endif
#ifdef GPM_LAYER_SYM_EVEN	// For even number of interneurons
int	VectSize = (H_NUMRAYS*NUMINTER + H_NUMINTER*NUMINTER + H_NUMINTER*NUMMOTOR) + 3*(H_NUMINTER+2);
#endif

// ------------------------------------
// Genotype-Phenotype Mapping Function
// ------------------------------------
// Unconstrained, Fully recurrent
#ifdef GPM_FULL
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Weights of the neural network
	for (int i = 1; i <= CIRCUITSIZE; i++){
		for (int j = 1; j <= CIRCUITSIZE; j++){
			phen(k) = MapSearchParameter( gen(k), -WEIGHTMAX, WEIGHTMAX);
			k++;
		}
	}
	// Biases
	for (int i = 1; i <= CIRCUITSIZE; i++){
		phen(k) = MapSearchParameter( gen(k), -BIASMAX, BIASMAX);
		k++;
	}
	// TimeConstants
	for (int i = 1; i <= CIRCUITSIZE; i++){
		phen(k) = MapSearchParameter( gen(k), TAUMIN, TAUMAX);
		k++;
	}
	// Gains
	for (int i = 1; i <= CIRCUITSIZE; i++){
		phen(k) = MapSearchParameter( gen(k), GAINMIN, GAINMAX);
		k++;
	}
}
#endif

// Unconstrained, Sensory layer, Interneuron layer, Motorneuron layer
// Same one bias, timeconstant, and gain for all sensory neurons; Same one b, t, g, for the two motoneurons
#ifdef GPM_LAYER
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Weights from Sensor to Inter
	for (int i = 1; i <= NUMRAYS; i++){
		for (int j = 1; j <= NUMINTER; j++){
			phen(k) = MapSearchParameter( gen(k), -WEIGHTMAX, WEIGHTMAX);
			k++;
		}
	}
	// Weights from Inter to Inter
	for (int i = 1; i <= NUMINTER; i++){
		for (int j = 1; j <= NUMINTER; j++){
			phen(k) = MapSearchParameter( gen(k), -WEIGHTMAX, WEIGHTMAX);
			k++;
		}
	}
	// Weights from Inter to Motor
	for (int i = 1; i <= NUMINTER; i++){
		for (int j = 1; j <= NUMMOTOR; j++){
			phen(k) = MapSearchParameter( gen(k), -WEIGHTMAX, WEIGHTMAX);
			k++;
		}
	}
	// Biases: Interneurons + One for sensor and one for motor
	for (int i = 1; i <= NUMINTER+2; i++){
		phen(k) = MapSearchParameter( gen(k), -BIASMAX, BIASMAX);
		k++;
	}
	// TimeConstants: Interneurons + One for sensor and one for motor
	for (int i = 1; i <= NUMINTER+2; i++){
		phen(k) = MapSearchParameter( gen(k), TAUMIN, TAUMAX);
		k++;
	}
	// Gains: Interneurons + One for sensor and one for motor
	for (int i = 1; i <= NUMINTER+2; i++){
		phen(k) = MapSearchParameter( gen(k), GAINMIN, GAINMAX);
		k++;
	}
}
#endif

// For odd number of interneurons
// Left-right symmetric, Sensory layer, Interneuron layer, Motorneuron layer
#ifdef GPM_LAYER_SYM_ODD
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Weights from Sensor to Inter
	for (int i = 1; i <= H_NUMRAYS - 1; i++){
		for (int j = 1; j <= NUMINTER; j++){
			phen(k) = MapSearchParameter( gen(k), -WEIGHTMAX, WEIGHTMAX);
			k++;
		}
	}
	for (int j = 1; j <= H_NUMINTER; j++){
		phen(k) = MapSearchParameter( gen(k), -WEIGHTMAX, WEIGHTMAX);
		k++;
	}
	// Weights from Inter to Inter
	for (int i = 1; i <= H_NUMINTER-1; i++){
		for (int j = 1; j <= NUMINTER; j++){
			phen(k) = MapSearchParameter( gen(k), -WEIGHTMAX, WEIGHTMAX);
			k++;
		}
	}
	for (int j = 1; j <= H_NUMINTER; j++){
		phen(k) = MapSearchParameter( gen(k), -WEIGHTMAX, WEIGHTMAX);
		k++;
	}
	// Weights from Inter to Motor
	for (int i = 1; i <= H_NUMINTER-1; i++){
		for (int j = 1; j <= NUMMOTOR; j++){
			phen(k) = MapSearchParameter( gen(k), -WEIGHTMAX, WEIGHTMAX);
			k++;
		}
	}
	for (int j = 1; j <= H_NUMMOTOR; j++){
		phen(k) = MapSearchParameter( gen(k), -WEIGHTMAX, WEIGHTMAX);
		k++;
	}
	// Biases: Interneurons + One for sensor and one for motor
	for (int i = 1; i <= H_NUMINTER+1; i++){
		phen(k) = MapSearchParameter( gen(k), -BIASMAX, BIASMAX);
		k++;
	}
	// TimeConstants: Interneurons + One for sensor and one for motor
	for (int i = 1; i <= H_NUMINTER+1; i++){
		phen(k) = MapSearchParameter( gen(k), TAUMIN, TAUMAX);
		k++;
	}
	// Gains: Interneurons + One for sensor and one for motor
	for (int i = 1; i <= H_NUMINTER+1; i++){
		phen(k) = MapSearchParameter( gen(k), GAINMIN, GAINMAX);
		k++;
	}
}
#endif

// For even number of interneurons
// Left-right symmetric, Sensory layer, Interneuron layer, Motorneuron layer
#ifdef GPM_LAYER_SYM_EVEN
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Weights from Sensor to Inter
	for (int i = 1; i <= H_NUMRAYS; i++){
		for (int j = 1; j <= NUMINTER; j++){
			phen(k) = MapSearchParameter( gen(k), -WEIGHTMAX, WEIGHTMAX);
			k++;
		}
	}
	// Weights from Inter to Inter
	for (int i = 1; i <= H_NUMINTER; i++){
		for (int j = 1; j <= NUMINTER; j++){
			phen(k) = MapSearchParameter( gen(k), -WEIGHTMAX, WEIGHTMAX);
			k++;
		}
	}
	// Weights from Inter to Motor
	for (int i = 1; i <= H_NUMINTER; i++){
		for (int j = 1; j <= NUMMOTOR; j++){
			phen(k) = MapSearchParameter( gen(k), -WEIGHTMAX, WEIGHTMAX);
			k++;
		}
	}
	// Biases: Interneurons + One for sensor and one for motor
	for (int i = 1; i <= H_NUMINTER+2; i++){
		phen(k) = MapSearchParameter( gen(k), -BIASMAX, BIASMAX);
		k++;
	}
	// TimeConstants: Interneurons + One for sensor and one for motor
	for (int i = 1; i <= H_NUMINTER+2; i++){
		phen(k) = MapSearchParameter( gen(k), TAUMIN, TAUMAX);
		k++;
	}
	// Gains: Interneurons + One for sensor and one for motor
	for (int i = 1; i <= H_NUMINTER+2; i++){
		phen(k) = MapSearchParameter( gen(k), GAINMIN, GAINMAX);
		k++;
	}
}
#endif

// ------------------------------------
// Fitness Functions
// ------------------------------------

// - - - - - - - - - - - - - - - - - -
// NSF RI 2019
// Task 1: Perceiving Affordances using Lines as walls
// 		Gaps between the lines larger than self should be approached through.
//    Gaps between the lines smaller than self should be avoided.
// - - - - - - - - - - - - - - - - - -
double PerceiveAffordance(TVector<double> &v, RandomState &rs)
{
	double fit = 0.0;
	int trials = 0;
	double final_distance = 0.0;
	VisualAgent Agent(0.0,0.0,NUMRAYS,NUMINTER,NUMMOTOR);
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(v, phenotype);
	Agent.SetController(phenotype);
	Line ObjectLeft(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
	Line ObjectRight(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
	for (double pos = MINPOS; pos <= MAXPOS; pos += POSSTEP)
	{
		if (pos != 0)
		{
			for (double gapsize = MINSIZE; gapsize <= MAXSIZE; gapsize += SIZESTEP)
			{
				if (gapsize != BodySize)
				{
					for (double reps = 0; reps <= REPS; reps += 1)
					{
						Agent.Reset(0, 0, 0);
						Agent.SetPositionX(0);
						ObjectRight.SetPositionX((gapsize/2)+(OBJECTHEIGHT/2.0) + pos);
						ObjectLeft.SetPositionX((-(gapsize/2)-(OBJECTHEIGHT/2.0)) + pos);
						ObjectRight.SetPositionY(STARTHEIGHT);
						ObjectLeft.SetPositionY(STARTHEIGHT);
						for (double t = 0; ObjectLeft.PositionY() > BodySize; t += StepSize) {
							Agent.Step2(rs, StepSize, ObjectLeft, ObjectRight);
							ObjectLeft.Step(StepSize);
							ObjectRight.Step(StepSize);
						}
						final_distance = fabs(Agent.PositionX() - pos); //XXX
						if (gapsize < BodySize){
							final_distance = final_distance > MAXDISTANCE ? 1.0 : final_distance/MAXDISTANCE;
						}
						else {
							final_distance = final_distance > MAXDISTANCE ? 0.0 : (MAXDISTANCE - final_distance)/MAXDISTANCE;
						}
						trials += 1;
						fit += final_distance;
					}
				}
			}
		}
	}
	fit = fit/trials;
	return fit;
}

void BehaviorPA(TVector<double> &v, RandomState &rs)
{
	ofstream avoid_posx,approach_posx;
	ofstream avoid_n1,avoid_n2,avoid_n3,avoid_n4,avoid_n5,avoid_n6,avoid_n7;
	// ofstream avoid_n8,avoid_n9,avoid_n10,avoid_n11,avoid_n12,avoid_n13,avoid_n14,avoid_n15;
	ofstream approach_n1,approach_n2,approach_n3,approach_n4,approach_n5,approach_n6,approach_n7;
	// ofstream approach_n8,approach_n9,approach_n10,approach_n11,approach_n12,approach_n13,approach_n14,approach_n15;
	ofstream avoid_s1,avoid_s2,avoid_s3,avoid_s4,avoid_s5,avoid_s6,avoid_s7,avoid_s8,avoid_s9,avoid_s10,avoid_s11,avoid_s12,avoid_s13,avoid_s14,avoid_s15;
	ofstream approach_s1,approach_s2,approach_s3,approach_s4,approach_s5,approach_s6,approach_s7,approach_s8,approach_s9,approach_s10,approach_s11,approach_s12,approach_s13,approach_s14,approach_s15;
	ofstream avoid_m1,avoid_m2;
	ofstream approach_m1,approach_m2;

	avoid_posx.open("A_avoid_aposx.dat");
	avoid_n1.open("A_avoid_n1.dat");
	avoid_n2.open("A_avoid_n2.dat");
	avoid_n3.open("A_avoid_n3.dat");
	avoid_n4.open("A_avoid_n4.dat");
	avoid_n5.open("A_avoid_n5.dat");
	avoid_n6.open("A_avoid_n6.dat");
	avoid_n7.open("A_avoid_n7.dat");
	// avoid_n8.open("A_avoid_n8.dat");
	// avoid_n9.open("A_avoid_n9.dat");
	// avoid_n10.open("A_avoid_n10.dat");
	// avoid_n11.open("A_avoid_n11.dat");
	// avoid_n12.open("A_avoid_n12.dat");
	// avoid_n13.open("A_avoid_n13.dat");
	// avoid_n14.open("A_avoid_n14.dat");
	// avoid_n15.open("A_avoid_n15.dat");

	approach_posx.open("A_approach_aposx.dat");
	approach_n1.open("A_approach_n1.dat");
	approach_n2.open("A_approach_n2.dat");
	approach_n3.open("A_approach_n3.dat");
	approach_n4.open("A_approach_n4.dat");
	approach_n5.open("A_approach_n5.dat");
	approach_n6.open("A_approach_n6.dat");
	approach_n7.open("A_approach_n7.dat");
	// approach_n8.open("A_approach_n8.dat");
	// approach_n9.open("A_approach_n9.dat");
	// approach_n10.open("A_approach_n10.dat");
	// approach_n11.open("A_approach_n11.dat");
	// approach_n12.open("A_approach_n12.dat");
	// approach_n13.open("A_approach_n13.dat");
	// approach_n14.open("A_approach_n14.dat");
	// approach_n15.open("A_approach_n15.dat");

	avoid_s1.open("A_avoid_s1.dat");
	avoid_s2.open("A_avoid_s2.dat");
	avoid_s3.open("A_avoid_s3.dat");
	avoid_s4.open("A_avoid_s4.dat");
	avoid_s5.open("A_avoid_s5.dat");
	avoid_s6.open("A_avoid_s6.dat");
	avoid_s7.open("A_avoid_s7.dat");
	avoid_s8.open("A_avoid_s8.dat");
	avoid_s9.open("A_avoid_s9.dat");
	avoid_s10.open("A_avoid_s10.dat");
	avoid_s11.open("A_avoid_s11.dat");
	avoid_s12.open("A_avoid_s12.dat");
	avoid_s13.open("A_avoid_s13.dat");
	avoid_s14.open("A_avoid_s14.dat");
	avoid_s15.open("A_avoid_s15.dat");

	avoid_m1.open("A_avoid_m1.dat");
	avoid_m2.open("A_avoid_m2.dat");

	approach_s1.open("A_approach_s1.dat");
	approach_s2.open("A_approach_s2.dat");
	approach_s3.open("A_approach_s3.dat");
	approach_s4.open("A_approach_s4.dat");
	approach_s5.open("A_approach_s5.dat");
	approach_s6.open("A_approach_s6.dat");
	approach_s7.open("A_approach_s7.dat");
	approach_s8.open("A_approach_s8.dat");
	approach_s9.open("A_approach_s9.dat");
	approach_s10.open("A_approach_s10.dat");
	approach_s11.open("A_approach_s11.dat");
	approach_s12.open("A_approach_s12.dat");
	approach_s13.open("A_approach_s13.dat");
	approach_s14.open("A_approach_s14.dat");
	approach_s15.open("A_approach_s15.dat");

	approach_m1.open("A_approach_m1.dat");
	approach_m2.open("A_approach_m2.dat");

	//--
	double fit = 0.0;
	int trials = 0;
	double final_distance = 0.0;
	VisualAgent Agent(0.0,0.0,NUMRAYS,NUMINTER,NUMMOTOR);
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(v, phenotype);
	Agent.SetController(phenotype);
	Line ObjectLeft(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
	Line ObjectRight(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
	for (double pos = MINPOS; pos <= MAXPOS; pos += POSSTEP)
	{
		if (pos != 0)
		{
			for (double gapsize = MINSIZE; gapsize <= MAXSIZE; gapsize += SIZESTEP)
			{
				if (gapsize != BodySize)
				{
					for (double reps = 0; reps <= REPS; reps += 1)
					{
						Agent.Reset(0, 0, 0);
						Agent.SetPositionX(0);
						ObjectRight.SetPositionX((gapsize/2)+(OBJECTHEIGHT/2.0) + pos);
						ObjectLeft.SetPositionX((-(gapsize/2)-(OBJECTHEIGHT/2.0)) + pos);
						ObjectRight.SetPositionY(STARTHEIGHT);
						ObjectLeft.SetPositionY(STARTHEIGHT);
						for (double t = 0; ObjectLeft.PositionY() > BodySize; t += StepSize) {
							Agent.Step2(rs, StepSize, ObjectLeft, ObjectRight);
							ObjectLeft.Step(StepSize);
							ObjectRight.Step(StepSize);
							if (gapsize < BodySize){
								avoid_posx << Agent.PositionX() - pos << " ";
								avoid_n1 << Agent.NervousSystem.NeuronOutput(1) << " ";
								avoid_n2 << Agent.NervousSystem.NeuronOutput(2) << " ";
								avoid_n3 << Agent.NervousSystem.NeuronOutput(3) << " ";
								avoid_n4 << Agent.NervousSystem.NeuronOutput(4) << " ";
								avoid_n5 << Agent.NervousSystem.NeuronOutput(5) << " ";
								avoid_n6 << Agent.NervousSystem.NeuronOutput(6) << " ";
								avoid_n7 << Agent.NervousSystem.NeuronOutput(7) << " ";
								// avoid_n8 << Agent.NervousSystem.NeuronOutput(8) << " ";
								// avoid_n9 << Agent.NervousSystem.NeuronOutput(9) << " ";
								// avoid_n10 << Agent.NervousSystem.NeuronOutput(10) << " ";
								// avoid_n11 << Agent.NervousSystem.NeuronOutput(11) << " ";
								// avoid_n12 << Agent.NervousSystem.NeuronOutput(12) << " ";
								// avoid_n13 << Agent.NervousSystem.NeuronOutput(13) << " ";
								// avoid_n14 << Agent.NervousSystem.NeuronOutput(14) << " ";
								// avoid_n15 << Agent.NervousSystem.NeuronOutput(15) << " ";

								avoid_s1 << Agent.ExternalInput(1) << " ";
								avoid_s2 << Agent.ExternalInput(2) << " ";
								avoid_s3 << Agent.ExternalInput(3) << " ";
								avoid_s4 << Agent.ExternalInput(4) << " ";
								avoid_s5 << Agent.ExternalInput(5) << " ";
								avoid_s6 << Agent.ExternalInput(6) << " ";
								avoid_s7 << Agent.ExternalInput(7) << " ";
								avoid_s8 << Agent.ExternalInput(8) << " ";
								avoid_s9 << Agent.ExternalInput(9) << " ";
								avoid_s10 << Agent.ExternalInput(10) << " ";
								avoid_s11 << Agent.ExternalInput(11) << " ";
								avoid_s12 << Agent.ExternalInput(12) << " ";
								avoid_s13 << Agent.ExternalInput(13) << " ";
								avoid_s14 << Agent.ExternalInput(14) << " ";
								avoid_s15 << Agent.ExternalInput(15) << " ";

								avoid_m1 << Agent.NervousSystem.NeuronOutput(8) << " ";
								avoid_m2 << Agent.NervousSystem.NeuronOutput(9) << " ";
							}
							else {
								approach_posx << Agent.PositionX() - pos << " ";
								approach_n1 << Agent.NervousSystem.NeuronOutput(1) << " ";
								approach_n2 << Agent.NervousSystem.NeuronOutput(2) << " ";
								approach_n3 << Agent.NervousSystem.NeuronOutput(3) << " ";
								approach_n4 << Agent.NervousSystem.NeuronOutput(4) << " ";
								approach_n5 << Agent.NervousSystem.NeuronOutput(5) << " ";
								approach_n6 << Agent.NervousSystem.NeuronOutput(6) << " ";
								approach_n7 << Agent.NervousSystem.NeuronOutput(7) << " ";
								// approach_n8 << Agent.NervousSystem.NeuronOutput(8) << " ";
								// approach_n9 << Agent.NervousSystem.NeuronOutput(9) << " ";
								// approach_n10 << Agent.NervousSystem.NeuronOutput(10) << " ";
								// approach_n11 << Agent.NervousSystem.NeuronOutput(11) << " ";
								// approach_n12 << Agent.NervousSystem.NeuronOutput(12) << " ";
								// approach_n13 << Agent.NervousSystem.NeuronOutput(13) << " ";
								// approach_n14 << Agent.NervousSystem.NeuronOutput(14) << " ";
								// approach_n15 << Agent.NervousSystem.NeuronOutput(15) << " ";

								approach_s1 << Agent.ExternalInput(1) << " ";
								approach_s2 << Agent.ExternalInput(2) << " ";
								approach_s3 << Agent.ExternalInput(3) << " ";
								approach_s4 << Agent.ExternalInput(4) << " ";
								approach_s5 << Agent.ExternalInput(5) << " ";
								approach_s6 << Agent.ExternalInput(6) << " ";
								approach_s7 << Agent.ExternalInput(7) << " ";
								approach_s8 << Agent.ExternalInput(8) << " ";
								approach_s9 << Agent.ExternalInput(9) << " ";
								approach_s10 << Agent.ExternalInput(10) << " ";
								approach_s11 << Agent.ExternalInput(11) << " ";
								approach_s12 << Agent.ExternalInput(12) << " ";
								approach_s13 << Agent.ExternalInput(13) << " ";
								approach_s14 << Agent.ExternalInput(14) << " ";
								approach_s15 << Agent.ExternalInput(15) << " ";

								approach_m1 << Agent.NervousSystem.NeuronOutput(8) << " ";
								approach_m2 << Agent.NervousSystem.NeuronOutput(9) << " ";
							}
						}
						final_distance = fabs(Agent.PositionX() - pos);
						if (gapsize < BodySize){
							final_distance = final_distance > MAXDISTANCE ? 1.0 : final_distance/MAXDISTANCE;
							avoid_posx << endl;
							avoid_n1 << endl;
							avoid_n2 << endl;
							avoid_n3 << endl;
							avoid_n4 << endl;
							avoid_n5 << endl;
							avoid_n6 << endl;
							avoid_n7 << endl;
							// avoid_n8 << endl;
							// avoid_n9 << endl;
							// avoid_n10 << endl;
							// avoid_n11 << endl;
							// avoid_n12 << endl;
							// avoid_n13 << endl;
							// avoid_n14 << endl;
							// avoid_n15 << endl;

							avoid_s1 << endl;
							avoid_s2 << endl;
							avoid_s3 << endl;
							avoid_s4 << endl;
							avoid_s5 << endl;
							avoid_s6 << endl;
							avoid_s7 << endl;
							avoid_s8 << endl;
							avoid_s9 << endl;
							avoid_s10 << endl;
							avoid_s11 << endl;
							avoid_s12 << endl;
							avoid_s13 << endl;
							avoid_s14 << endl;
							avoid_s15 << endl;

							avoid_m1 << endl;
							avoid_m2 << endl;
						}
						else {
							final_distance = final_distance > MAXDISTANCE ? 0.0 : (MAXDISTANCE - final_distance)/MAXDISTANCE;
							approach_posx << endl;
							approach_n1 << endl;
							approach_n2 << endl;
							approach_n3 << endl;
							approach_n4 << endl;
							approach_n5 << endl;
							approach_n6 << endl;
							approach_n7 << endl;
							// approach_n8 << endl;
							// approach_n9 << endl;
							// approach_n10 << endl;
							// approach_n11 << endl;
							// approach_n12 << endl;
							// approach_n13 << endl;
							// approach_n14 << endl;
							// approach_n15 << endl;

							approach_s1 << endl;
							approach_s2 << endl;
							approach_s3 << endl;
							approach_s4 << endl;
							approach_s5 << endl;
							approach_s6 << endl;
							approach_s7 << endl;
							approach_s8 << endl;
							approach_s9 << endl;
							approach_s10 << endl;
							approach_s11 << endl;
							approach_s12 << endl;
							approach_s13 << endl;
							approach_s14 << endl;
							approach_s15 << endl;

							approach_m1 << endl;
							approach_m2 << endl;
						}
						trials += 1;
						fit += final_distance;
					}
				}
			}
		}
	}
	fit = fit/trials;
	cout << "Fitness: " << fit << endl;

	avoid_posx.close();
	avoid_n1.close();
	avoid_n2.close();
	avoid_n3.close();
	avoid_n4.close();
	avoid_n5.close();
	avoid_n6.close();
	avoid_n7.close();
	// avoid_n8.close();
	// avoid_n9.close();
	// avoid_n10.close();
	// avoid_n11.close();
	// avoid_n12.close();
	// avoid_n13.close();
	// avoid_n14.close();
	// avoid_n15.close();

	avoid_s1.close();
	avoid_s2.close();
	avoid_s3.close();
	avoid_s4.close();
	avoid_s5.close();
	avoid_s6.close();
	avoid_s7.close();
	avoid_s8.close();
	avoid_s9.close();
	avoid_s10.close();
	avoid_s11.close();
	avoid_s12.close();
	avoid_s13.close();
	avoid_s14.close();
	avoid_s15.close();

	avoid_m1.close();
	avoid_m2.close();

	approach_posx.close();
	approach_n1.close();
	approach_n2.close();
	approach_n3.close();
	approach_n4.close();
	approach_n5.close();
	approach_n6.close();
	approach_n7.close();
	// approach_n8.close();
	// approach_n9.close();
	// approach_n10.close();
	// approach_n11.close();
	// approach_n12.close();
	// approach_n13.close();
	// approach_n14.close();
	// approach_n15.close();

	approach_s1.close();
	approach_s2.close();
	approach_s3.close();
	approach_s4.close();
	approach_s5.close();
	approach_s6.close();
	approach_s7.close();
	approach_s8.close();
	approach_s9.close();
	approach_s10.close();
	approach_s11.close();
	approach_s12.close();
	approach_s13.close();
	approach_s14.close();
	approach_s15.close();

	approach_m1.close();
	approach_m2.close();
}

void GeneralizationPA(TVector<double> &v, RandomState &rs)
{
	ofstream finaloffset,finalperf;
	finaloffset.open("A_finaloffset.dat");
	finalperf.open("A_finalperf.dat");
	//--
	double fit = 0.0;
	int trials = 0;
	double final_distance = 0.0;
	VisualAgent Agent(0.0,0.0,NUMRAYS,NUMINTER,NUMMOTOR);
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(v, phenotype);
	Agent.SetController(phenotype);
	Line ObjectLeft(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
	Line ObjectRight(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
	for (double pos = 1.0; pos <= 5.0; pos += 0.01)
	{
		for (double gapsize = 20.0; gapsize <= 40.0; gapsize += 0.05)
		{
			Agent.Reset(0, 0, 0);
			Agent.SetPositionX(0);
			ObjectRight.SetPositionX((gapsize/2)+(OBJECTHEIGHT/2.0) + pos);
			ObjectLeft.SetPositionX((-(gapsize/2)-(OBJECTHEIGHT/2.0)) + pos);
			ObjectRight.SetPositionY(STARTHEIGHT);
			ObjectLeft.SetPositionY(STARTHEIGHT);
			for (double t = 0; ObjectLeft.PositionY() > BodySize; t += StepSize) {
				Agent.Step2(rs, StepSize, ObjectLeft, ObjectRight);
				ObjectLeft.Step(StepSize);
				ObjectRight.Step(StepSize);
			}
			final_distance = fabs(Agent.PositionX() - pos);
			finaloffset << final_distance << " ";
			if (gapsize < BodySize){
				final_distance = final_distance > MAXDISTANCE ? 1.0 : final_distance/MAXDISTANCE;
			}
			else {
				final_distance = final_distance > MAXDISTANCE ? 0.0 : (MAXDISTANCE - final_distance)/MAXDISTANCE;
			}
			trials += 1;
			fit += final_distance;
		}
		finaloffset << endl;
	}
	fit = fit/trials;
	finalperf << fit << endl;
	finaloffset.close();
	finalperf.close();
}

void ConnectionLesionsPA(TVector<double> &v, RandomState &rs)
{
	ofstream ftotal("lesionsA.dat"),fapproach("lesionsA_approach.dat"),favoid("lesionsA_avoid.dat");
	for (int from = 1; from <= NUMINTER; from++)
	{
		for (int to = 1; to <= NUMINTER; to++)
		{
			cout << from << ", " << to << endl;
			double fit = 0.0, fit_avoid = 0.0, fit_approach = 0.0;
			int trials = 0, trials_approach = 0, trials_avoid = 0;
			double final_distance = 0.0;
			VisualAgent Agent(0.0,0.0,NUMRAYS,NUMINTER,NUMMOTOR);
			TVector<double> phenotype;
			phenotype.SetBounds(1, VectSize);
			GenPhenMapping(v, phenotype);
			Agent.SetController(phenotype);
			Agent.NervousSystem.SetConnectionWeight(from,to,0.0); 		// Delete connection
			Line ObjectLeft(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
			Line ObjectRight(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
			for (double pos = MINPOS; pos <= MAXPOS; pos += 1.0)
			{
				if (pos != 0)
				{
					for (double gapsize = MINSIZE; gapsize <= MAXSIZE; gapsize += 1.0)
					{
						if (gapsize != BodySize)
						{
							for (double reps = 0; reps <= REPS; reps += 1)
							{
								Agent.Reset(0, 0, 0);
								Agent.SetPositionX(0);
								ObjectRight.SetPositionX((gapsize/2)+(OBJECTHEIGHT/2.0) + pos);
								ObjectLeft.SetPositionX((-(gapsize/2)-(OBJECTHEIGHT/2.0)) + pos);
								ObjectRight.SetPositionY(STARTHEIGHT);
								ObjectLeft.SetPositionY(STARTHEIGHT);
								for (double t = 0; ObjectLeft.PositionY() > BodySize; t += StepSize) {
									Agent.Step2(rs, StepSize, ObjectLeft, ObjectRight);
									ObjectLeft.Step(StepSize);
									ObjectRight.Step(StepSize);
								}
								final_distance = fabs(Agent.PositionX() - pos); //XXX
								if (gapsize < BodySize){
									final_distance = final_distance > MAXDISTANCE ? 1.0 : final_distance/MAXDISTANCE;
									fit_avoid += final_distance;
									trials_avoid += 1;
								}
								else {
									final_distance = final_distance > MAXDISTANCE ? 0.0 : (MAXDISTANCE - final_distance)/MAXDISTANCE;
									fit_approach += final_distance;
									trials_approach += 1;
								}
								trials += 1;
								fit += final_distance;
							}
						}
					}
				}
			}
			fit = fit/trials;
			ftotal << fit << " ";
			fapproach << fit_approach/trials_approach << " ";
			favoid << fit_avoid/trials_avoid << " ";
		}
		ftotal << endl;
		fapproach << endl;
		favoid << endl;
	}
	ftotal.close();
	fapproach.close();
	favoid.close();
}

void InformationalLesionsPA(TVector<double> &v, RandomState &rs)
{
	ofstream ftotal("infolesionsA.dat"),fapproach("infolesionsA_approach.dat"),favoid("infolesionsA_avoid.dat");
	for (int from = 1; from <= NUMINTER; from++)
	{
		for (int to = 1; to <= NUMINTER; to++)
		{
			cout << from << ", " << to << endl;
			double fit = 0.0, fit_avoid = 0.0, fit_approach = 0.0;
			int trials = 0, trials_approach = 0, trials_avoid = 0;
			double final_distance = 0.0;
			VisualAgent Agent(0.0,0.0,NUMRAYS,NUMINTER,NUMMOTOR);
			TVector<double> phenotype;
			phenotype.SetBounds(1, VectSize);
			GenPhenMapping(v, phenotype);
			Agent.SetController(phenotype);
			Agent.NervousSystem.SetConnectionWeight(from,to,0.0); 		// Delete connection
			Line ObjectLeft(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
			Line ObjectRight(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
			for (double pos = MINPOS; pos <= MAXPOS; pos += 1.0)
			{
				if (pos != 0)
				{
					for (double gapsize = MINSIZE; gapsize <= MAXSIZE; gapsize += 1.0)
					{
						if (gapsize != BodySize)
						{
							for (double reps = 0; reps <= REPS; reps += 1)
							{
								Agent.Reset(0, 0, 0);
								Agent.SetPositionX(0);
								ObjectRight.SetPositionX((gapsize/2)+(OBJECTHEIGHT/2.0) + pos);
								ObjectLeft.SetPositionX((-(gapsize/2)-(OBJECTHEIGHT/2.0)) + pos);
								ObjectRight.SetPositionY(STARTHEIGHT);
								ObjectLeft.SetPositionY(STARTHEIGHT);
								for (double t = 0; ObjectLeft.PositionY() > BodySize; t += StepSize) {
									Agent.Step2(rs, StepSize, ObjectLeft, ObjectRight);
									ObjectLeft.Step(StepSize);
									ObjectRight.Step(StepSize);
								}
								final_distance = fabs(Agent.PositionX() - pos); //XXX
								if (gapsize < BodySize){
									final_distance = final_distance > MAXDISTANCE ? 1.0 : final_distance/MAXDISTANCE;
									fit_avoid += final_distance;
									trials_avoid += 1;
								}
								else {
									final_distance = final_distance > MAXDISTANCE ? 0.0 : (MAXDISTANCE - final_distance)/MAXDISTANCE;
									fit_approach += final_distance;
									trials_approach += 1;
								}
								trials += 1;
								fit += final_distance;
							}
						}
					}
				}
			}
			fit = fit/trials;
			ftotal << fit << " ";
			fapproach << fit_approach/trials_approach << " ";
			favoid << fit_avoid/trials_avoid << " ";
		}
		ftotal << endl;
		fapproach << endl;
		favoid << endl;
	}
	ftotal.close();
	fapproach.close();
	favoid.close();
}
// - - - - - - - - - - - - - - - - - -
// NSF RI 2019
// Task 2: Size categorization using Circles.
// 		Circles smaller than self should be caught.
//		Circles larger than self should be avoided.
// - - - - - - - - - - - - - - - - - -
double CircleSizeCategorization(TVector<double> &v, RandomState &rs)
{
	double fit = 0.0;
	int trials = 0;
	double final_distance = 0.0;
	VisualAgent Agent(0.0,0.0,NUMRAYS,NUMINTER,NUMMOTOR);
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(v, phenotype);
	Agent.SetController(phenotype);
	Circle Object(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
	for (double pos = MINPOS; pos <= MAXPOS; pos += POSSTEP)
	{
		if (pos != 0)
		{
			for (double size = MINSIZE; size <= MAXSIZE; size += SIZESTEP)
			{
				if (size != BodySize)
				{
					for (double reps = 0; reps <= REPS; reps += 1)
					{
						Agent.Reset(0, 0, 0);
						Agent.SetPositionX(0);
						Object.SetPositionY(STARTHEIGHT);
						Object.SetPositionX(pos);
						Object.SetSize(size);
						for (double t = 0; Object.PositionY() > BodySize; t += StepSize) {
							Agent.Step(rs, StepSize, Object);
							Object.Step(StepSize);
						}
						final_distance = fabs(Agent.PositionX() - Object.PositionX());
						if (size < BodySize){
							final_distance = final_distance > MAXDISTANCE ? 0.0 : (MAXDISTANCE - final_distance)/MAXDISTANCE;
						}
						else {
							final_distance = final_distance > MAXDISTANCE ? 1.0 : final_distance/MAXDISTANCE;
						}
						trials += 1;
						fit += final_distance;
					}
				}
			}
		}
	}
	fit = fit/trials;
	return fit;
}

void BehaviorCC(TVector<double> &v, RandomState &rs)
{
	ofstream avoid_posx,approach_posx;
	ofstream avoid_n1,avoid_n2,avoid_n3,avoid_n4,avoid_n5,avoid_n6,avoid_n7;
	// ofstream avoid_n8,avoid_n9,avoid_n10,avoid_n11,avoid_n12,avoid_n13,avoid_n14,avoid_n15;
	ofstream approach_n1,approach_n2,approach_n3,approach_n4,approach_n5,approach_n6,approach_n7;
	// ofstream approach_n8,approach_n9,approach_n10,approach_n11,approach_n12,approach_n13,approach_n14,approach_n15;
	ofstream avoid_s1,avoid_s2,avoid_s3,avoid_s4,avoid_s5,avoid_s6,avoid_s7,avoid_s8,avoid_s9,avoid_s10,avoid_s11,avoid_s12,avoid_s13,avoid_s14,avoid_s15;
	ofstream approach_s1,approach_s2,approach_s3,approach_s4,approach_s5,approach_s6,approach_s7,approach_s8,approach_s9,approach_s10,approach_s11,approach_s12,approach_s13,approach_s14,approach_s15;
	ofstream avoid_m1,avoid_m2;
	ofstream approach_m1,approach_m2;

	avoid_posx.open("B_avoid_aposx.dat");
	avoid_n1.open("B_avoid_n1.dat");
	avoid_n2.open("B_avoid_n2.dat");
	avoid_n3.open("B_avoid_n3.dat");
	avoid_n4.open("B_avoid_n4.dat");
	avoid_n5.open("B_avoid_n5.dat");
	avoid_n6.open("B_avoid_n6.dat");
	avoid_n7.open("B_avoid_n7.dat");
	// avoid_n8.open("B_avoid_n8.dat");
	// avoid_n9.open("B_avoid_n9.dat");
	// avoid_n10.open("B_avoid_n10.dat");
	// avoid_n11.open("B_avoid_n11.dat");
	// avoid_n12.open("B_avoid_n12.dat");
	// avoid_n13.open("B_avoid_n13.dat");
	// avoid_n14.open("B_avoid_n14.dat");
	// avoid_n15.open("B_avoid_n15.dat");

	approach_posx.open("B_approach_aposx.dat");
	approach_n1.open("B_approach_n1.dat");
	approach_n2.open("B_approach_n2.dat");
	approach_n3.open("B_approach_n3.dat");
	approach_n4.open("B_approach_n4.dat");
	approach_n5.open("B_approach_n5.dat");
	approach_n6.open("B_approach_n6.dat");
	approach_n7.open("B_approach_n7.dat");
	// approach_n8.open("B_approach_n8.dat");
	// approach_n9.open("B_approach_n9.dat");
	// approach_n10.open("B_approach_n10.dat");
	// approach_n11.open("B_approach_n11.dat");
	// approach_n12.open("B_approach_n12.dat");
	// approach_n13.open("B_approach_n13.dat");
	// approach_n14.open("B_approach_n14.dat");
	// approach_n15.open("B_approach_n15.dat");

	avoid_s1.open("B_avoid_s1.dat");
	avoid_s2.open("B_avoid_s2.dat");
	avoid_s3.open("B_avoid_s3.dat");
	avoid_s4.open("B_avoid_s4.dat");
	avoid_s5.open("B_avoid_s5.dat");
	avoid_s6.open("B_avoid_s6.dat");
	avoid_s7.open("B_avoid_s7.dat");
	avoid_s8.open("B_avoid_s8.dat");
	avoid_s9.open("B_avoid_s9.dat");
	avoid_s10.open("B_avoid_s10.dat");
	avoid_s11.open("B_avoid_s11.dat");
	avoid_s12.open("B_avoid_s12.dat");
	avoid_s13.open("B_avoid_s13.dat");
	avoid_s14.open("B_avoid_s14.dat");
	avoid_s15.open("B_avoid_s15.dat");

	avoid_m1.open("B_avoid_m1.dat");
	avoid_m2.open("B_avoid_m2.dat");

	approach_s1.open("B_approach_s1.dat");
	approach_s2.open("B_approach_s2.dat");
	approach_s3.open("B_approach_s3.dat");
	approach_s4.open("B_approach_s4.dat");
	approach_s5.open("B_approach_s5.dat");
	approach_s6.open("B_approach_s6.dat");
	approach_s7.open("B_approach_s7.dat");
	approach_s8.open("B_approach_s8.dat");
	approach_s9.open("B_approach_s9.dat");
	approach_s10.open("B_approach_s10.dat");
	approach_s11.open("B_approach_s11.dat");
	approach_s12.open("B_approach_s12.dat");
	approach_s13.open("B_approach_s13.dat");
	approach_s14.open("B_approach_s14.dat");
	approach_s15.open("B_approach_s15.dat");

	approach_m1.open("B_approach_m1.dat");
	approach_m2.open("B_approach_m2.dat");
	//--
	double fit = 0.0;
	int trials = 0;
	double final_distance = 0.0;
	VisualAgent Agent(0.0,0.0,NUMRAYS,NUMINTER,NUMMOTOR);
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(v, phenotype);
	Agent.SetController(phenotype);

	ofstream sweights,iweights,mweights;
	sweights.open("sensorweights.dat");
	iweights.open("interweights.dat");
	mweights.open("motorweights.dat");
	for (int i = 1; i <= NUMRAYS; i += 1)
	{
		for (int j = 1; j <= NUMINTER; j += 1)
		{
			sweights << Agent.SensorWeight(i,j) << " ";
		}
		sweights << endl;
	}
	for (int i = 1; i <= NUMINTER; i += 1)
	{
		for (int j = 1; j <= NUMINTER; j += 1)
		{
			iweights << Agent.NervousSystem.ConnectionWeight(i,j) << " ";
		}
		iweights << endl;
	}
	for (int i = 1; i <= NUMINTER; i += 1)
	{
		for (int j = NUMINTER+1; j <= NUMINTER + NUMMOTOR; j += 1)
		{
			mweights << Agent.NervousSystem.ConnectionWeight(i,j) << " ";
		}
		mweights << endl;
	}
	sweights.close();
	iweights.close();
	mweights.close();

	Circle Object(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
	for (double pos = MINPOS; pos <= MAXPOS; pos += POSSTEP)
	{
		if (pos != 0)
		{
			for (double size = MINSIZE; size <= MAXSIZE; size += SIZESTEP)
			{
				if (size != BodySize)
				{
					for (double reps = 0; reps <= REPS; reps += 1)
					{
						Agent.Reset(0, 0, 0);
						Agent.SetPositionX(0);
						Object.SetPositionY(STARTHEIGHT);
						Object.SetPositionX(pos);
						Object.SetSize(size);
						for (double t = 0; Object.PositionY() > BodySize; t += StepSize) {
							Agent.Step(rs, StepSize, Object);
							Object.Step(StepSize);
							if (size < BodySize){
								approach_posx << Agent.PositionX() - Object.PositionX() << " ";
								approach_n1 << Agent.NervousSystem.NeuronOutput(1) << " ";
								approach_n2 << Agent.NervousSystem.NeuronOutput(2) << " ";
								approach_n3 << Agent.NervousSystem.NeuronOutput(3) << " ";
								approach_n4 << Agent.NervousSystem.NeuronOutput(4) << " ";
								approach_n5 << Agent.NervousSystem.NeuronOutput(5) << " ";
								approach_n6 << Agent.NervousSystem.NeuronOutput(6) << " ";
								approach_n7 << Agent.NervousSystem.NeuronOutput(7) << " ";
								// approach_n8 << Agent.NervousSystem.NeuronOutput(8) << " ";
								// approach_n9 << Agent.NervousSystem.NeuronOutput(9) << " ";
								// approach_n10 << Agent.NervousSystem.NeuronOutput(10) << " ";
								// approach_n11 << Agent.NervousSystem.NeuronOutput(11) << " ";
								// approach_n12 << Agent.NervousSystem.NeuronOutput(12) << " ";
								// approach_n13 << Agent.NervousSystem.NeuronOutput(13) << " ";
								// approach_n14 << Agent.NervousSystem.NeuronOutput(14) << " ";
								// approach_n15 << Agent.NervousSystem.NeuronOutput(15) << " ";

								approach_s1 << Agent.ExternalInput(1) << " ";
								approach_s2 << Agent.ExternalInput(2) << " ";
								approach_s3 << Agent.ExternalInput(3) << " ";
								approach_s4 << Agent.ExternalInput(4) << " ";
								approach_s5 << Agent.ExternalInput(5) << " ";
								approach_s6 << Agent.ExternalInput(6) << " ";
								approach_s7 << Agent.ExternalInput(7) << " ";
								approach_s8 << Agent.ExternalInput(8) << " ";
								approach_s9 << Agent.ExternalInput(9) << " ";
								approach_s10 << Agent.ExternalInput(10) << " ";
								approach_s11 << Agent.ExternalInput(11) << " ";
								approach_s12 << Agent.ExternalInput(12) << " ";
								approach_s13 << Agent.ExternalInput(13) << " ";
								approach_s14 << Agent.ExternalInput(14) << " ";
								approach_s15 << Agent.ExternalInput(15) << " ";

								approach_m1 << Agent.NervousSystem.NeuronOutput(8) << " ";
								approach_m2 << Agent.NervousSystem.NeuronOutput(9) << " ";
							}
							else {
								avoid_posx << Agent.PositionX() - Object.PositionX() << " ";
								avoid_n1 << Agent.NervousSystem.NeuronOutput(1) << " ";
								avoid_n2 << Agent.NervousSystem.NeuronOutput(2) << " ";
								avoid_n3 << Agent.NervousSystem.NeuronOutput(3) << " ";
								avoid_n4 << Agent.NervousSystem.NeuronOutput(4) << " ";
								avoid_n5 << Agent.NervousSystem.NeuronOutput(5) << " ";
								avoid_n6 << Agent.NervousSystem.NeuronOutput(6) << " ";
								avoid_n7 << Agent.NervousSystem.NeuronOutput(7) << " ";
								// avoid_n8 << Agent.NervousSystem.NeuronOutput(8) << " ";
								// avoid_n9 << Agent.NervousSystem.NeuronOutput(9) << " ";
								// avoid_n10 << Agent.NervousSystem.NeuronOutput(10) << " ";
								// avoid_n11 << Agent.NervousSystem.NeuronOutput(11) << " ";
								// avoid_n12 << Agent.NervousSystem.NeuronOutput(12) << " ";
								// avoid_n13 << Agent.NervousSystem.NeuronOutput(13) << " ";
								// avoid_n14 << Agent.NervousSystem.NeuronOutput(14) << " ";
								// avoid_n15 << Agent.NervousSystem.NeuronOutput(15) << " ";

								avoid_s1 << Agent.ExternalInput(1) << " ";
								avoid_s2 << Agent.ExternalInput(2) << " ";
								avoid_s3 << Agent.ExternalInput(3) << " ";
								avoid_s4 << Agent.ExternalInput(4) << " ";
								avoid_s5 << Agent.ExternalInput(5) << " ";
								avoid_s6 << Agent.ExternalInput(6) << " ";
								avoid_s7 << Agent.ExternalInput(7) << " ";
								avoid_s8 << Agent.ExternalInput(8) << " ";
								avoid_s9 << Agent.ExternalInput(9) << " ";
								avoid_s10 << Agent.ExternalInput(10) << " ";
								avoid_s11 << Agent.ExternalInput(11) << " ";
								avoid_s12 << Agent.ExternalInput(12) << " ";
								avoid_s13 << Agent.ExternalInput(13) << " ";
								avoid_s14 << Agent.ExternalInput(14) << " ";
								avoid_s15 << Agent.ExternalInput(15) << " ";

								avoid_m1 << Agent.NervousSystem.NeuronOutput(8) << " ";
								avoid_m2 << Agent.NervousSystem.NeuronOutput(9) << " ";
							}
						}
						final_distance = fabs(Agent.PositionX() - Object.PositionX());
						if (size < BodySize){
							final_distance = final_distance > MAXDISTANCE ? 0.0 : (MAXDISTANCE - final_distance)/MAXDISTANCE;
							approach_posx << endl;
							approach_n1 << endl;
							approach_n2 << endl;
							approach_n3 << endl;
							approach_n4 << endl;
							approach_n5 << endl;
							approach_n6 << endl;
							approach_n7 << endl;
							// approach_n8 << endl;
							// approach_n9 << endl;
							// approach_n10 << endl;
							// approach_n11 << endl;
							// approach_n12 << endl;
							// approach_n13 << endl;
							// approach_n14 << endl;
							// approach_n15 << endl;

							approach_s1 << endl;
							approach_s2 << endl;
							approach_s3 << endl;
							approach_s4 << endl;
							approach_s5 << endl;
							approach_s6 << endl;
							approach_s7 << endl;
							approach_s8 << endl;
							approach_s9 << endl;
							approach_s10 << endl;
							approach_s11 << endl;
							approach_s12 << endl;
							approach_s13 << endl;
							approach_s14 << endl;
							approach_s15 << endl;


							approach_m1 << endl;
							approach_m2 << endl;
						}
						else {
							final_distance = final_distance > MAXDISTANCE ? 1.0 : final_distance/MAXDISTANCE;
							avoid_posx << endl;
							avoid_n1 << endl;
							avoid_n2 << endl;
							avoid_n3 << endl;
							avoid_n4 << endl;
							avoid_n5 << endl;
							avoid_n6 << endl;
							avoid_n7 << endl;
							// avoid_n8 << endl;
							// avoid_n9 << endl;
							// avoid_n10 << endl;
							// avoid_n11 << endl;
							// avoid_n12 << endl;
							// avoid_n13 << endl;
							// avoid_n14 << endl;
							// avoid_n15 << endl;

							avoid_s1 << endl;
							avoid_s2 << endl;
							avoid_s3 << endl;
							avoid_s4 << endl;
							avoid_s5 << endl;
							avoid_s6 << endl;
							avoid_s7 << endl;
							avoid_s8 << endl;
							avoid_s9 << endl;
							avoid_s10 << endl;
							avoid_s11 << endl;
							avoid_s12 << endl;
							avoid_s13 << endl;
							avoid_s14 << endl;
							avoid_s15 << endl;

							avoid_m1 << endl;
							avoid_m2 << endl;
						}
						trials += 1;
						fit += final_distance;
					}
				}
			}
		}
	}
	fit = fit/trials;
	cout << "Fitness: " << fit << endl;

	avoid_posx.close();
	avoid_n1.close();
	avoid_n2.close();
	avoid_n3.close();
	avoid_n4.close();
	avoid_n5.close();
	avoid_n6.close();
	avoid_n7.close();
	// avoid_n8.close();
	// avoid_n9.close();
	// avoid_n10.close();
	// avoid_n11.close();
	// avoid_n12.close();
	// avoid_n13.close();
	// avoid_n14.close();
	// avoid_n15.close();
	avoid_s1.close();
	avoid_s2.close();
	avoid_s3.close();
	avoid_s4.close();
	avoid_s5.close();
	avoid_s6.close();
	avoid_s7.close();
	avoid_s8.close();
	avoid_s9.close();
	avoid_s10.close();
	avoid_s11.close();
	avoid_s12.close();
	avoid_s13.close();
	avoid_s14.close();
	avoid_s15.close();

	avoid_m1.close();
	avoid_m2.close();

	approach_posx.close();
	approach_n1.close();
	approach_n2.close();
	approach_n3.close();
	approach_n4.close();
	approach_n5.close();
	approach_n6.close();
	approach_n7.close();
	// approach_n8.close();
	// approach_n9.close();
	// approach_n10.close();
	// approach_n11.close();
	// approach_n12.close();
	// approach_n13.close();
	// approach_n14.close();
	// approach_n15.close();
	approach_s1.close();
	approach_s2.close();
	approach_s3.close();
	approach_s4.close();
	approach_s5.close();
	approach_s6.close();
	approach_s7.close();
	approach_s8.close();
	approach_s9.close();
	approach_s10.close();
	approach_s11.close();
	approach_s12.close();
	approach_s13.close();
	approach_s14.close();
	approach_s15.close();

	approach_m1.close();
	approach_m2.close();
}

void GeneralizationCC(TVector<double> &v, RandomState &rs)
{
	ofstream finaloffset,finalperf;
	finaloffset.open("B_finaloffset.dat");
	finalperf.open("B_finalperf.dat");
	//--
	double fit = 0.0;
	int trials = 0;
	double final_distance = 0.0;
	VisualAgent Agent(0.0,0.0,NUMRAYS,NUMINTER,NUMMOTOR);
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(v, phenotype);
	Agent.SetController(phenotype);
	Circle Object(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
	for (double pos = 1.0; pos <= 5.0; pos += 0.01)
	{
		for (double size = 20.0; size <= 40.0; size += 0.05)
		{
			Agent.Reset(0, 0, 0);
			Agent.SetPositionX(0);
			Object.SetPositionY(STARTHEIGHT);
			Object.SetPositionX(pos);
			Object.SetSize(size);
			for (double t = 0; Object.PositionY() > BodySize; t += StepSize) {
				Agent.Step(rs, StepSize, Object);
				Object.Step(StepSize);
			}
			final_distance = fabs(Agent.PositionX() - Object.PositionX());
			finaloffset << final_distance << " ";
			if (size < BodySize){
				final_distance = final_distance > MAXDISTANCE ? 0.0 : (MAXDISTANCE - final_distance)/MAXDISTANCE;
			}
			else {
				final_distance = final_distance > MAXDISTANCE ? 1.0 : final_distance/MAXDISTANCE;
			}
			trials += 1;
			fit += final_distance;
		}
		finaloffset << endl;
	}
	fit = fit/trials;
	finalperf << fit << endl;
	finaloffset.close();
	finalperf.close();
}

void ConnectionLesionsCC(TVector<double> &v, RandomState &rs)
{
	ofstream ftotal("lesionsB.dat"),fapproach("lesionsB_approach.dat"),favoid("lesionsB_avoid.dat");
	for (int from = 1; from <= NUMINTER; from++)
	{
		for (int to = 1; to <= NUMINTER; to++)
		{
			cout << from << ", " << to << endl;
			double fit = 0.0, fit_avoid = 0.0, fit_approach = 0.0;
			int trials = 0, trials_approach = 0, trials_avoid = 0;
			double final_distance = 0.0;
			VisualAgent Agent(0.0,0.0,NUMRAYS,NUMINTER,NUMMOTOR);
			TVector<double> phenotype;
			phenotype.SetBounds(1, VectSize);
			GenPhenMapping(v, phenotype);
			Agent.SetController(phenotype);
			Agent.NervousSystem.SetConnectionWeight(from,to,0.0); 		// Delete connection
			Circle Object(0.0,STARTHEIGHT,-3,0.0,OBJECTHEIGHT);
			for (double pos = MINPOS; pos <= MAXPOS; pos += 1.0)
			{
				if (pos != 0)
				{
					for (double size = MINSIZE; size <= MAXSIZE; size += 1.0)
					{
						if (size != BodySize)
						{
							for (double reps = 0; reps <= REPS; reps += 1)
							{
								Agent.Reset(0, 0, 0);
								Agent.SetPositionX(0);
								Object.SetPositionY(STARTHEIGHT);
								Object.SetPositionX(pos);
								Object.SetSize(size);
								for (double t = 0; Object.PositionY() > BodySize; t += StepSize) {
									Agent.Step(rs, StepSize, Object);
									Object.Step(StepSize);
								}
								final_distance = fabs(Agent.PositionX() - Object.PositionX());
								if (size < BodySize){
									final_distance = final_distance > MAXDISTANCE ? 0.0 : (MAXDISTANCE - final_distance)/MAXDISTANCE;
									fit_approach += final_distance;
									trials_approach += 1;
								}
								else {
									final_distance = final_distance > MAXDISTANCE ? 1.0 : final_distance/MAXDISTANCE;
									fit_avoid += final_distance;
									trials_avoid += 1;
								}
								trials += 1;
								fit += final_distance;
							}
						}
					}
				}
			}
			fit = fit/trials;
			ftotal << fit << " ";
			fapproach << fit_approach/trials_approach << " ";
			favoid << fit_avoid/trials_avoid << " ";
		}
		ftotal << endl;
		fapproach << endl;
		favoid << endl;
	}
	ftotal.close();
	fapproach.close();
	favoid.close();
}

// - - - - - - - - - - - - - - - - - -
// NSF RI 2019
// Multiple tasks
// - - - - - - - - - - - - - - - - - -
double MultipleTasks(TVector<double> &v, RandomState &rs)
{
	double fit1 = PerceiveAffordance(v, rs);
	double fit2 = CircleSizeCategorization(v, rs);
	return fit1 * fit2;
}
// ------------------------------------
// Display
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
}

void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	ofstream BestIndividualFile;

	bestVector = s.BestIndividual();
	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile << setprecision(32);
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();
}

int TerminationFunction(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	if (BestPerf > 0.95) return 1;
	else return 0;
}

// ------------------------------------
// Main
// ------------------------------------
#ifdef EVOLVE
int main (int argc, const char* argv[]) {
	TSearch s(VectSize);

	// Configure the search
	long seed = static_cast<long>(time(NULL));

	// save the seed to a file
	ofstream seedfile;
	seedfile.open ("seed.dat");
	seedfile << seed << endl;
	seedfile.close();

	s.SetRandomSeed(seed);
	s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
	s.SetSearchResultsDisplayFunction(ResultsDisplay);
	s.SetSelectionMode(RANK_BASED);
	s.SetReproductionMode(GENETIC_ALGORITHM);
	s.SetPopulationSize(POPSIZE); //200
	s.SetMaxGenerations(GENS);  //100
	s.SetMutationVariance(0.01); //8.0
	s.SetCrossoverProbability(0.5);
	s.SetCrossoverMode(UNIFORM);
	s.SetMaxExpectedOffspring(1.1);
	s.SetElitistFraction(0.1);
	s.SetSearchConstraint(1);
	s.SetCheckpointInterval(0);
	// s.SetReEvaluationFlag(1);

	// redirect standard output to a file
	#ifdef PRINTTOFILE
	ofstream evolfile;
	evolfile.open("fitness.dat");
	cout.rdbuf(evolfile.rdbuf());
	#endif
	// TASK 2: Object-size Categorization
	s.SetSearchTerminationFunction(NULL);
	//s.SetEvaluationFunction(CircleSizeCategorization);//B
	//s.SetEvaluationFunction(PerceiveAffordance); //A
	s.SetEvaluationFunction(MultipleTasks);//C
	s.ExecuteSearch();

	#ifdef PRINTTOFILE
	evolfile.close();
	#endif

	return 0;
}
#else
int main (int argc, const char* argv[])
{
	RandomState rs;
	long seed = static_cast<long>(time(NULL));
	rs.SetRandomSeed(seed);
	std::cout << std::setprecision(10);
	ifstream BestIndividualFile;
	TVector<double> bestVector(1, VectSize);
	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile >> bestVector;
	// BehaviorCC(bestVector, rs);
	// BehaviorPA(bestVector, rs);
	// GeneralizationCC(bestVector, rs);
	// GeneralizationPA(bestVector, rs);
	ConnectionLesionsCC(bestVector, rs);
	ConnectionLesionsPA(bestVector, rs);
	return 0;
}
#endif
