Point(1) = {0.000000, 0.000000, 0.000000, 0.012500};
Point(2) = {0.000000, -0.050000, 0.000000, 0.00500};
Point(3) = {0.000000, -0.065000, 0.000000, 0.00500};
Point(4) = {7.000000, -0.065000, 0.000000, 0.00500};
Point(5) = {7.000000, -0.050000, 0.000000, 0.00500};
Point(6) = {7.000000, 0.000000, 0.000000, 0.012500};
Point(7) = {0.000000, -0.565000, 0.000000, 0.125000};
Point(8) = {7.000000, -0.565000, 0.000000, 0.125000};
Line(1) = {1, 2};
Line(2) = {2, 5};
Line(3) = {5, 6};
Line(4) = {6, 1};
Line(5) = {3, 2};
Line(6) = {4, 3};
Line(7) = {5, 4};
Line(8) = {8, 4};
Line(9)	= {3, 7};
Line(10) = {7, 8};
Line Loop(11) = {1, 2, 3, 4};
Line Loop(12) = {5, 2, 7, 6};
Line Loop(13) = {8, 6, 9, 10};
Plane Surface(1) = {11};
Plane Surface(2) = {12};
Plane Surface(3) = {13};

