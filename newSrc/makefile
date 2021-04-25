main: main.o CTRNN.o TSearch.o VisualAgent.o random.o
	g++ -pthread -o main main.o CTRNN.o TSearch.o VisualAgent.o random.o
random.o: random.cpp random.h VectorMatrix.h
	g++ -c -O3 -flto random.cpp
CTRNN.o: CTRNN.cpp random.h CTRNN.h
	g++ -c -O3 -flto CTRNN.cpp
TSearch.o: TSearch.cpp TSearch.h
	g++ -c -O3 -flto TSearch.cpp
VisualAgent.o: VisualAgent.cpp VisualAgent.h VisualObject.h CTRNN.h random.h VectorMatrix.h
	g++ -c -O3 -flto VisualAgent.cpp
main.o: main.cpp CTRNN.h VisualAgent.h TSearch.h
	g++ -c -O3 -flto main.cpp
clean:
	rm *.o main
