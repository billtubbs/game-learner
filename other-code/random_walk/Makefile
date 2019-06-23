CXX = g++
CFLAGS = -O3
LIBS = -lg++
PROG = walk

$(PROG):	main_walk.cc Walk.o
		$(CXX) $(CFLAGS) $^ -o $@ $(LIBS)

Walk.o:		Walk.cc
		$(CXX) $(CFLAGS) -c $^

test:		on-line off-line
		gnuplot 'walk.gnu'

on-line:
		$(PROG) 19 on 0 1.0 0.05 1
		$(PROG) 19 on 0 1.0 0.05 2
		$(PROG) 19 on 0 1.0 0.05 3
		$(PROG) 19 on 0 1.0 0.05 5
		$(PROG) 19 on 0 1.0 0.05 8
		$(PROG) 19 on 0 1.0 0.05 15
		$(PROG) 19 on 0 1.0 0.05 30
		$(PROG) 19 on 0 1.0 0.05 60
		$(PROG) 19 on 0 1.0 0.05 100
		$(PROG) 19 on 0 1.0 0.05 200
		$(PROG) 19 on 0 1.0 0.05 1000
off-line:
		$(PROG) 19 off 0 0.3 0.05 1
		$(PROG) 19 off 0 0.3 0.05 2
		$(PROG) 19 off 0 0.3 0.05 3
		$(PROG) 19 off 0 0.3 0.05 4
		$(PROG) 19 off 0 0.3 0.05 6
		$(PROG) 19 off 0 0.3 0.05 8
		$(PROG) 19 off 0 0.3 0.05 15
		$(PROG) 19 off 0 0.3 0.05 30
		$(PROG) 19 off 0 0.3 0.05 60
		$(PROG) 19 off 0 0.3 0.05 100
		$(PROG) 19 off 0 0.3 0.05 200
		$(PROG) 19 off 0 0.3 0.05 1000
clean:
		rm -f walk Walk.o

