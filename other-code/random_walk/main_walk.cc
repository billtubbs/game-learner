/* -*- C++ -*- */
//
// main_walk.cc,
//
//     run series of random walks

#include "Walk.h"
#include <fstream.h>
#include <stdlib.h>
#include <stdio.h>

#define N_EXPERIMENTS 100
#define N_EPISODES    10 

inline double sqr(double x) { return x*x; } 

inline double rms_error(const Array &a, const Array &b)
{
  if(a.size() != b.size()) {
    cerr << "rms_error error: incompatible dimensions" << endl;
    exit(1);
  } else if (a.size() < 3) {
    cerr << "rms_error error: too small" << endl;
    exit(1);
  }

  double error = 0;
  for(int i=1; i<a.size()-1; i++) {
    error += sqr(a[i] - b[i]);
  }
  error = sqrt(error / (a.size()-2) );
  return error;
}

main(int argc, char **argv)
{
  int 
    size,              // size of random walk
    on_off,            // 1 is on-line, 0 is off-line
    n_step;

  double 
    rms_1,             // cumulative root mean square
    rms_2,             // temporary root mean square
    a_start,
    a_end,
    a_inc;

  char name[256];      // store file name

  if(argc != 7) {
      cerr << "usage: walk" << endl
	   << "\t<size>\t\tnumber: size of random walk" << endl
	   << "\t<update>\ton,off: on-line or off-line updates" << endl
	   << "\t<start_alpha>\tnumber: tinitial alpha parameter" << endl
	   << "\t<final_alpha>\tnumber: final alpha parameter" << endl
	   << "\t<alpha_step>\tnumber: step size (increment)" << endl
	   << "\t<n_step>\tnumber: number of steps in walk" << endl;
      exit(1);
  }
  size = atoi(argv[1]) + 2;  // add two for rewards on left and right
  on_off = (!strcmp(argv[2], "on")) ? 1 : 0 ;
  a_start = atof(argv[3]);
  a_end = atof(argv[4]);
  a_inc = atof(argv[5]);
  n_step = atoi(argv[6]);

  // find correct (DP) probabilities to within small theta 
  Walk ans(size, -1.0, 1.0);
  ans.solve(0.00000001, 1.0);
  cout << "Correct (approx) values (DP solution)" << endl;
  cout << ans << endl;

  // find the n-step TD probabilities
  Walk walk(size, -1.0, 1.0);

  walk.reset();
  sprintf(name, "graph_%d_%s.plot", n_step, argv[2]);
  ofstream graph(name);

  graph.setf(ios::fixed, ios::floatfield);
  cout.setf(ios::fixed, ios::floatfield);
  for(double a=a_start; a<=a_end; a += a_inc) {
      cout.precision(6);
      cout << "\talpha " << a << "\t";

      rms_2 = 0;
      for(int expe=0; expe < N_EXPERIMENTS; expe++) {
	  rms_1 = 0;
	  walk = 0.0;
	  for(int episode=0; episode < N_EPISODES; episode++) {
	      walk.update(size/2,n_step, 1.0, a, (bool)on_off);
	      rms_1 += rms_error(ans, walk);
	  }
	  rms_1 /= (double)N_EPISODES;
	  rms_2 += rms_1;
      }
      rms_2 /= (double)N_EXPERIMENTS;
      cout.precision(12);
      cout << rms_2 << endl;
      graph.precision(12);
      graph << rms_2 << endl;
  }

  exit(0);
}

