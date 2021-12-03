#!/usr/bin/env python 
from mpi4py import MPI
import numpy
import sys

#Transpose of a matrix using MPI 

def main(*args, **kwargs):
    rows = int(args[1])
    cols = int(args[2])

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #Initiate the matrix
    if rank==0:
        matrix = numpy.matrix([[1.,1.,1.,1.,1.],[5.,5.,5.,5.,5.],[9.,9.,9.,9.,9.]])
        #matrix = numpy.random.randint(10, size=(rows, cols)).astype("float")
        print("\n[The Original Matrix]\n{}".format(matrix))
        pass
    else:
        matrix = None

    #Effort to Synchronize all.
    comm.Barrier()
    
    #Create an empty row vector with the length of columns.
    matrixbuf = numpy.zeros(cols)
    comm.Scatter(matrix, matrixbuf, root=0)
    print("[process {}] received {}".format(rank, matrixbuf))

    #Effort to Synchronize all.
    comm.Barrier()
    
    #Create an empty column vector and store the transpose in it.
    transposed = matrixbuf[numpy.newaxis, :].T        
    results = comm.gather(transposed,root=0)

    #Rank 0 releases the result.
    if rank == 0:
       print("\n[The Transposed Matrix]\n{}".format(numpy.column_stack(results)))

#Actually this runs the program
if __name__ == '__main__':
    sys.exit(main(*sys.argv))
