/* -*-C++-*- */
//
// Walk.cc,
//
//     random walk method definitions

#include "Walk.h"

void Walk::solve(double theta, double gam)
{
  double v, delta = 0;

  values[left] = value(left);
  values[right] = value(right);

  do {
    delta = 0;
    for(int i=1; i<sz-1; i++) {
      v = values[i];
      values[i] =  0.5 * ( gam * (values[i-1] + values[i+1]) );
      delta = max(delta, fabs(v-values[i]));
    }
  } while ( delta > theta );
}

void Walk::update(int state, int n, double gam, double alpha, bool on)
{
  Pix nxt;
  double v;
  int s;

  path.clear();
  ranwalk(state); // create an episode

  if(on) {                                    // on-line updates
    for(Pix i = path.first(); i; path.next(i)) {
      if(done(s = path(i)))
	values[s] = reward(s);
      else {
	v = values[s];
	nxt = i; path.next(nxt);
	values[s] += (alpha * (step(nxt, n, gam) - v));
      }
    }
  } else {                                   // off-line updates
    Array deltas(size()); deltas = 0;             
    for(Pix i = path.first(); i; path.next(i)) {
      if(done(s = path(i)))
	values[s] = reward(s);
      else {
	v = values[s];
	nxt = i; path.next(nxt);
	deltas[s] += (alpha * (step(nxt, n, gam) - v));
      }
    }
    (*this) += deltas;
  }
}

double Walk::step(Pix i, int n, double gam)
{
  if ( done(path(i)) ) {
    return ( reward(path(i)) );
  } else if ( n == 1 ) {
    return ( reward(path(i)) + gam * values[path(i)] );
  }  else {
    Pix pt = i; path.next(pt);
    return ( reward(path(i)) + gam * step(pt, n-1, gam) );
  }
}

double Walk::ranwalk(int state)
{
  //  cout << "(" << state << ") ";
  path.append(state);
  
  if( done(state) ) {
    return (reward(state));
  } else {
    return ranwalk( next(state) );
  }
}



