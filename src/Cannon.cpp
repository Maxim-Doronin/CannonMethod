//#include <omp.h>
#include <iostream>
#include <ctime> 
using namespace std;

inline int idx(int i, int j, int size) {
	return i*size + j;
}

int matrixGenerate(double *&M, const int size) {
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			M[idx(i, j, size)] = rand() % 1000;
	return 0;
}

int matrixPrint(double *&M, const int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++)
			cout << M[idx(i, j, size)] << "\t";
		cout << endl;
	}
	cout << endl;
	return 0;
}

int main(int argc, char** argv) {
	double *A = nullptr;
	double *B = nullptr;
	double *C = nullptr;
	int size = 0;
	
	size = atoi(argv[1]);

	srand((unsigned int)time(NULL));
	A = new double [size*size];
	B = new double [size*size];
	matrixGenerate(A, size);
	matrixGenerate(B, size);
	
	matrixPrint(A, size);
	matrixPrint(B, size);





	delete []A;
	delete []B;
	return 0;
}