# laplace3d.plt
# Automatically generated gnuplot file to display the results of testing.
# 
# For PNG output, use something like the following:
# $ gnuplot
# gnuplot> set terminal png large size 800, 1200
# gnuplot> set output 'laplace3d.png'
# gnuplot> load 'laplace3d.plt'
# gnuplot> exit
# 
# Alternatively, uncomment the two lines below and run:
# $ gnuplot laplace3d.plt
# 

reset

#set terminal png large size 800, 1200
#set output 'laplace3d.png'


set datafile separator ","
set datafile missing ""

unset xlabel
set ylabel "Score"
unset key
set title "Tuning Results"



set xrange [0:10]
set yrange [0.535336112976:0.644843959808]

set format x ""

set xtics 1, 1, 9




set xtics nomirror
set ytics nomirror
set ytics out
set xtics in

set border 3


set lmargin 12





# MULTIPLOT
# The main graph above, smaller graphs below showing how the variables change.


set size 1,1
set origin 0,0
set multiplot





# Main graph gets 40%, labels, tics etc at bottom get 10%
# Remaining 50% is divided evenly between the variables.


set size 1,0.4
set origin 0,0.6


set bmargin 0

set boxwidth 1
set style fill solid 0.2
set grid front
unset grid


plot "-" using 1:2 with boxes lc rgb "black", \
     "-" using 1:2 with points pt 2 lc rgb "black", \
     "-" using 1:2 with points pt 2 lc rgb "black", \
     "-" using 1:2 with points pt 2 lc rgb "black"
1, 0.557332038879
2, 0.563832998276
3, 0.557667970657
4, 0.560631036758
5, 0.583298921585
6, 0.579205989838
7, 0.57518196106
8, 0.55509519577
9, 0.557513952255
e
1, 0.555634021759
2, 0.553163766861
3, 0.551789999008
4, 0.556128025055
5, 0.56112909317
6, 0.57411813736
7, 0.55575799942
8, 0.550980091095
9, 0.556720972061
e
1, 0.557332038879
2, 0.563832998276
3, 0.557667970657
4, 0.560631036758
5, 0.583298921585
6, 0.579205989838
7, 0.57518196106
8, 0.55509519577
9, 0.557513952255
e
1, 0.606596946716
2, 0.576740980148
3, 0.58301281929
4, 0.583374023438
5, 0.629199981689
6, 0.605352878571
7, 0.604052066803
8, 0.571743965149
9, 0.584871768951
e





set border 27
set tmargin 0
unset xtics
unset title
set grid ytics lt 1 lc rgb "#cccccc"
set grid layerdefault
set grid noxtics



# Plot the graph for variable BLOCK_X
set size 1, 0.252
set origin 0,0.348

set ylabel "BLOCK_X"
set yrange [0:4]
set ytics ("32" 1, "64" 2, "128" 3)



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 2
3, 3
4, 1
5, 2
6, 3
7, 1
8, 2
9, 3
e




# Plot the graph for variable BLOCK_Y
set size 1, 0.252
set origin 0,0.096

set ylabel "BLOCK_Y"
set yrange [0:4]
set ytics ("2" 1, "4" 2, "6" 3)



set xlabel "Test No."
set xtics out
set xtics nomirror
set format x


set xtics 1, 1, 9


#set bmargin



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 1
3, 1
4, 2
5, 2
6, 2
7, 3
8, 3
9, 3
e






unset multiplot
reset

