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
set yrange [0.534478473663:0.624418449402]

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
1, 0.579154968262
2, 0.552774906158
3, 0.55030798912
4, 0.557199954987
5, 0.555194139481
6, 0.58017706871
7, 0.554831027985
8, 0.554163217545
9, 0.558702945709
e
1, 0.552680015564
2, 0.550972938538
3, 0.549688100815
4, 0.554998874664
5, 0.552047967911
6, 0.57498383522
7, 0.547327041626
8, 0.550820112228
9, 0.552355051041
e
1, 0.579154968262
2, 0.552774906158
3, 0.55030798912
4, 0.557199954987
5, 0.555194139481
6, 0.58017706871
7, 0.554831027985
8, 0.554163217545
9, 0.558702945709
e
1, 0.611569881439
2, 0.573209047318
3, 0.570899009705
4, 0.57789516449
5, 0.582362174988
6, 0.590630054474
7, 0.584486961365
8, 0.57758808136
9, 0.585399866104
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

