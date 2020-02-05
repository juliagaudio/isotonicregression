package support;

import java.util.ArrayList;

import gurobi.GRBException;

public class SequentialOptimizer {
	
	private double[][] X;
	private double[] Y;
	private int n;
	private int d;
	private int k;
	private ArrayList<Pair> oppositePairs;
	
	public SequentialOptimizer(double[][] X, double[] Y, int n, int d, int k, ArrayList<Pair> oppositePairs) {
		this.X = X;
		this.Y = Y;
		this.n = n;
		this.d = d;
		this.k = k;
		this.oppositePairs = oppositePairs;
	}
	
	public int[] optimize() throws GRBException{
		int[] activeCoordinates = new int[this.d];
		for (int i = 0; i < this.k; i++){
			// Zero out coordinates that have already been found
			Optimizer optimizer = new Optimizer(this.X, this.Y, this.n, this.d, 1, this.oppositePairs);
			optimizer.setLockedCoordinates(activeCoordinates);
			
			// Call Optimizer with k set to 1
			optimizer.solve();
			
			// Determine active coordinate
			int coordinate = optimizer.topCoordinate();
			
			optimizer.disposeModel();//new
			activeCoordinates[coordinate] = 1;	
		}		
		return activeCoordinates;
	}
	
	public void printSupport(int[] support){
		for (int i = 0; i < this.d; i++){
			if (support[i] == 1){
				System.out.print(Integer.toString(i) + ", ");
			}
		}
	}
}
