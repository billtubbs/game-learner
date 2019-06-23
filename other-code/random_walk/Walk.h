// -*- c++ -*-
//
// Walk.h
//
//     A random walk task.
//
// David Hirvonen <dhirvonen@sarnoff.com>

#ifndef WALK_H
#define WALK_H

#include <math.h>
#include <string.h>
#include <ACG.h>
#include <Uniform.h>
#include <SLList.h>

#include <iostream.h>

class Array
{
  friend ostream& operator <<(ostream &, const Array &);
public:
  Array(int n) : sz(n), values(new double[n]) { 
    bzero(values,n*sizeof(double));
  }
  ~Array() { delete [] values; }
  int size() const { return sz; }
  double& operator [](int i) { return values[i]; }
  double& operator [](int i) const { return values[i]; }
  void operator = (double);
  void operator += (const Array&);
  void operator /= (double);
protected:
  int sz;
  double *values;
};

inline void
Array::operator += (const Array &w)
{
  if(w.size() != size()) { 
    cerr << "Array +=: array dimensions do not match" << endl;
    exit(0);
  }
  for(int i=0; i<size(); i++)  values[i] += w[i];
}

inline void
Array::operator = (double v)
{
  for(int i=0; i<size(); i++)  values[i] = v;
}

inline void
Array::operator /= (double v)
{
  for(int i=0; i<size(); i++)  values[i] /= v;
}

inline ostream& operator <<(ostream &os, const Array &w)
{
  for(int i=0; i<w.sz; i++)
    os << "[" << w.values[i] << "]";
  return os;
}

class Walk : public Array {

public:

  Walk(int n, double lrw_, double rrw_) 
    : Array(n), left(0), right(n-1), lrw(lrw_), rrw(rrw_),
      gen( new ACG(113,77) ), rnd( new Uniform(0,1,gen) ) 
  { values[left] = lrw_; values[right] = rrw_; }

  ~Walk() { delete gen; delete rnd; delete [] values; }

  int next(int s) { return( ((*rnd)() < 0.5) ? (s-1) : (s+1)); }
  int done(int state) { return ((state == left) || (state == right)); }
  double reward(int state) { 
    return( (done(state)) ? ((state == left) ? lrw : rrw) : 0 ); 
  }
  double value(int state) {
    return((done(state)) ? ((state == left) ? lrw:rrw):values[state]); 
  }
  void operator =(double v);

  double ranwalk(int); // perform a random walk
  void update(int, int, double, double, bool on = true);
  double step(Pix, int, double);
  void solve(double, double);
  void test(int);
  void reset() { gen->reset(); }

private:

  int left, right;
  double lrw, rrw; // values,  rewards for terminal states

  ACG *gen;
  Uniform *rnd; 
  SLList<int> path;
};

inline double max(double x, double y) { 
  return (x > y) ? x : y; 
}

inline void
Walk::operator = (double v)
{
  for(int i=0; i<size(); i++)  
    values[i] = v;
}

inline void 
Walk::test(int n)
{
  int lcnt = 0, rcnt = 0;

  for(int i=0; i<n; i++) {
    if((*rnd)() < 0.5) lcnt++;
    else rcnt ++;
  }
  cout << lcnt << " " << rcnt << endl;
}

#endif // WALK_H





