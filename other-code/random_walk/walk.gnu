# generate a plot of random walk RMS error as parameter of alpha


set xlabel "alpha"
set ylabel "RMS error, averaged over 10 episodes"
set term postscript color "Times-Roman" 24

set title "N-Step TD Learning Random Walk (off-line)"
set output "rms_walk_off.ps"      

plot [:][.25:.55]\
"graph_1_off.plot"  t "n = 1" w lines, \
"graph_2_off.plot"  t "n = 2" w lines, \
"graph_3_off.plot"  t "n = 3" w lines, \
"graph_4_off.plot"  t "n = 4" w lines, \
"graph_6_off.plot"  t "n = 6" w lines, \
"graph_8_off.plot"  t "n = 8" w lines, \
"graph_15_off.plot"  t "n = 15" w lines, \
"graph_30_off.plot"  t "n = 30" w lines, \
"graph_60_off.plot"  t "n = 60" w lines, \
"graph_100_off.plot"  t "n = 100" w lines, \
"graph_200_off.plot"  t "n = 200" w lines, \
"graph_1000_off.plot"  t "n = 1000" w lines

set title "N-Step TD Learning Random Walk (on-line)"
set output "rms_walk_on.ps"      

plot [:][.25:.55]\
"graph_1_on.plot"  t "n = 1" w lines, \
"graph_2_on.plot"  t "n = 2" w lines, \
"graph_3_on.plot"  t "n = 3" w lines, \
"graph_5_on.plot"  t "n = 5" w lines, \
"graph_8_on.plot"  t "n = 8" w lines, \
"graph_15_on.plot"  t "n = 15" w lines, \
"graph_30_on.plot"  t "n = 30" w lines, \
"graph_60_on.plot"  t "n = 60" w lines, \
"graph_100_on.plot"  t "n = 100" w lines, \
"graph_200_on.plot"  t "n = 200" w lines, \
"graph_1000_on.plot"  t "n = 1000" w lines

set term x11
replot
