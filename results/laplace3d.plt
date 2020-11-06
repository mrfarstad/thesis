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



set xrange [0:29]
set yrange [2.18324012756:2.24956374169]

set format x ""

set xtics 5, 5, 25
set xtics add ("" 1 1, "" 2 1, "" 3 1, "" 4 1)
set xtics add ("" 26 1, "" 27 1, "" 28 1)
set mxtics 5




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
1, 2.22788095474
2, 2.21433401108
3, 2.21047782898
4, 2.2069568634
5, 2.21265101433
6, 2.20401287079
7, 2.20648097992
8, 2.20424485207
9, 2.2023870945
10, 2.20805001259
11, 2.20370984077
12, 2.20116090775
13, 2.20091199875
14, 2.19684004784
15, 2.20409297943
16, 2.20218706131
17, 2.20284199715
18, 2.20247602463
19, 2.2113301754
20, 2.20591211319
21, 2.20599102974
22, 2.20701098442
23, 2.20122098923
24, 
25, 2.20353913307
26, 2.20564889908
27, 2.20812296867
28, 
e
1, 2.22077417374
2, 2.20621490479
3, 2.19967007637
4, 2.2054848671
5, 2.20759510994
6, 2.19727802277
7, 2.20370578766
8, 2.19784212112
9, 2.20031404495
10, 2.19596195221
11, 2.19671702385
12, 2.19271492958
13, 2.19982099533
14, 2.19372010231
15, 2.19576001167
16, 2.1999900341
17, 2.20046210289
18, 2.20187807083
19, 2.19778609276
20, 2.20157313347
21, 2.20084500313
22, 2.19784808159
23, 2.19790697098
24, 
25, 2.19663405418
26, 2.19400596619
27, 2.20471096039
28, 2.23021006584
e
1, 2.22788095474
2, 2.21433401108
3, 2.21047782898
4, 2.2069568634
5, 2.21265101433
6, 2.20401287079
7, 2.20648097992
8, 2.20424485207
9, 2.2023870945
10, 2.20805001259
11, 2.20370984077
12, 2.20116090775
13, 2.20091199875
14, 2.19684004784
15, 2.20409297943
16, 2.20218706131
17, 2.20284199715
18, 2.20247602463
19, 2.2113301754
20, 2.20591211319
21, 2.20599102974
22, 2.20701098442
23, 2.20122098923
24, 
25, 2.20353913307
26, 2.20564889908
27, 2.20812296867
28, 2.21336698532
e
1, 2.24008893967
2, 2.21523499489
3, 2.21603989601
4, 2.21466422081
5, 2.22084307671
6, 2.20518302917
7, 2.21099114418
8, 2.20599913597
9, 2.21841096878
10, 2.2144920826
11, 2.21163487434
12, 2.21690297127
13, 2.22276496887
14, 2.20242500305
15, 2.20462703705
16, 2.20870900154
17, 2.21790909767
18, 2.22125911713
19, 2.21359014511
20, 2.2123811245
21, 2.21962594986
22, 2.20875692368
23, 2.20907115936
24, 
25, 2.21308517456
26, 2.21145701408
27, 2.22162699699
28, 
e





set border 27
set tmargin 0
unset xtics
unset title
set grid ytics lt 1 lc rgb "#cccccc"
set grid layerdefault
set grid noxtics



# Plot the graph for variable BLOCK_X
set size 1, 0.231
set origin 0,0.369

set ylabel "BLOCK_X"
set yrange [0:7]
set ytics ("16" 1, "32" 2, "48" 3, "64" 4, "128" 5, "256" 6)



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 2
3, 3
4, 4
5, 5
6, 6
7, 1
8, 2
9, 3
10, 4
11, 5
12, 6
13, 1
14, 2
15, 3
16, 4
17, 5
18, 6
19, 1
20, 2
21, 3
22, 4
23, 5
24, 6
25, 1
26, 2
27, 3
28, 4
e




# Plot the graph for variable BLOCK_Y
set size 1, 0.198
set origin 0,0.171

set ylabel "BLOCK_Y"
set yrange [0:6]
set ytics ("1" 1, "2" 2, "4" 3, "8" 4, "16" 5)



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 1
3, 1
4, 1
5, 1
6, 1
7, 2
8, 2
9, 2
10, 2
11, 2
12, 2
13, 3
14, 3
15, 3
16, 3
17, 3
18, 3
19, 4
20, 4
21, 4
22, 4
23, 4
24, 4
25, 5
26, 5
27, 5
28, 5
e




# Plot the graph for variable BLOCK_Z
set size 1, 0.066
set origin 0,0.105

set ylabel "BLOCK_Z"
set yrange [0:2]
set ytics ("1" 1)



set xlabel "Test No."
set xtics out
set xtics nomirror
set format x


set xtics 5, 5, 25
set xtics add ("" 1 1, "" 2 1, "" 3 1, "" 4 1)
set xtics add ("" 26 1, "" 27 1, "" 28 1)
set mxtics 5


#set bmargin



plot "-" using 1:2 with points pt 2 lc rgb "black"
1, 1
2, 1
3, 1
4, 1
5, 1
6, 1
7, 1
8, 1
9, 1
10, 1
11, 1
12, 1
13, 1
14, 1
15, 1
16, 1
17, 1
18, 1
19, 1
20, 1
21, 1
22, 1
23, 1
24, 1
25, 1
26, 1
27, 1
28, 1
e






unset multiplot
reset

