LIBS = -pthread -lm -lboost_system -lboost_filesystem -lboost_date_time -lcaffe -lglog -lgflags -lprotobuf `pkg-config --libs opencv`
OBJS = main.o corners-data-wrapper.o corners-regressor.o solution.o descriptors.o
all: main

main: $(OBJS)
	g++ -o main $(OBJS) $(LIBS)
	
main.o: main.cpp
	g++ -c -o main.o main.cpp
	
corners-data-wrapper.o: corners-data-wrapper.cpp corners-data-wrapper.h
	g++ -c -o corners-data-wrapper.o corners-data-wrapper.cpp
	
corners-regressor.o: corners-regressor.cpp corners-regressor.h
	g++ -c -o corners-regressor.o corners-regressor.cpp	
	
solution.o: solution.cpp solution.h
	g++ -c -o solution.o solution.cpp	

descriptors.o: descriptors.cpp descriptors.h
	g++ -c -o descriptors.o descriptors.cpp 


