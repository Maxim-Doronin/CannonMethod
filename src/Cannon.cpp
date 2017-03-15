#include <omp.h>
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

int matrixMult(double *&A, double *&B, double *&C, int size) {
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++) {
			C[idx(i, j, size)] = 0;
			for (int k = 0; k < size; k++)
				C[idx(i, j, size)] += A[idx(i, k, size)] * B[idx(k, j, size)];
		}
	return 0;
}

int shiftLeft(double *&M, int size, const int blockSize, const int init) {
	double *aux = nullptr;
	aux = new double[size];
	int step = blockSize;

	for (int k = 0, int s = 0; k < size; k += blockSize, s++) {
		for (int i = k; i < (k + blockSize); i++) {
			if (init > 0)
				step = s * blockSize;
			for (int j = 0; j < size; j++)
				aux[j] = M[idx(i, ((j + step) % size), size)];
			for (int j = 0; j < size; j++)
				M[idx(i, j, size)] = aux[j];
		}
	}
}

int shiftUP(double *&M, int size, const int blockSize, const int init) {
	double *aux = nullptr;
	aux = new double[size];
	int step = blockSize;

	for (int k = 0, int s = 0; k < size; k += blockSize, s++) {
		for (int i = k; i < (k + blockSize); i++) {
			if (init > 0)
				step = s * blockSize;
			for (int j = 0; j < size; j++)
				aux[j] = M[idx(((j + step) % size), i, size)];
			for (int j = 0; j < size; j++)
				M[idx(j, i, size)] = aux[j];
		}
	}
}



int cannonPar(double *&A, double *&B, double *&C, int xblock) {
	for (int i = 0; i < xblock; i++) {
		multProcessPar();
		shiftLeft();
		shiftUP();		
	}
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
	C = new double [size*size];
	matrixGenerate(A, size);
	matrixGenerate(B, size);
	
	//matrixPrint(A, size);
	//matrixPrint(B, size);

	double startTime = omp_get_wtime();
	matrixMult(A, B, C, size);
	double endTime = omp_get_wtime();
	
	//matrixPrint(C, size);
	cout << "Time: " << endTime - startTime << endl;

	delete []A;
	delete []B;
	delete []C;
	return 0;
}