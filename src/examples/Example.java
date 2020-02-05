package examples;

import gurobi.GRBException;

import java.util.ArrayList;
import java.util.Random;

import support.Pair;

public class Example {
	
	// Example inputs
	private int train = 200;
	private int test = 100;
	private int d = 10;
	private int sparsity = 3;
	private double noiseLevel = 0.5;

	private double[][] trainX;
	private double[] trainY;
	private double[][] testX;
	private double[] testY;
	private ArrayList<Pair> oppositePairs;
	
	public Example(){
		
	}

	public void generateData(){
		// f is the function that takes value 1 if x0 + x1 + x3 >= 1.5 and takes value zero otherwise.
		// Below I have generated random training data.
		Random random = new Random();
		this.trainX = new double[this.train][this.d];
		for (int i = 0; i < this.train; i++){
			for (int j = 0; j < this.d; j++){
				this.trainX[i][j] = random.nextDouble();
			}
		}

		this.trainY = new double[this.train];
		for (int i = 0; i < this.train; i++){
			double noise = 2 * this.noiseLevel * (random.nextDouble() - 0.5);
			double input = this.trainX[i][0] + this.trainX[i][1] + this.trainX[i][3] + noise; // Noisy Input Model
			this.trainY[i] = input >= 1.5 ? 1 : 0;
		}

		this.testX = new double[this.test][this.d];
		for (int i = 0; i < this.test; i++){
			for (int j = 0; j < this.d; j++){
				this.testX[i][j] = random.nextDouble();
			}
		}
		
		this.testY = new double[this.test];
		for (int i = 0; i < this.test; i++){
			double input = this.testX[i][0] + this.testX[i][1] + this.testX[i][3]; // no noise this time 
			this.testY[i] = input >= 1.5 ? 1 : 0;
		}

		// An example
		Pair p1 = new Pair(1, 5);
		Pair p2 = new Pair(2, 3);
		this.oppositePairs = new ArrayList<Pair>();
		this.oppositePairs.add(p1);
		this.oppositePairs.add(p2);
	}
	
	public double accuracy(int[] estimate, double[] truth){
		int count = estimate.length;
		int correct = 0;
		for (int i = 0; i < count; i++){
			correct += this.close(estimate[i], truth[i]) ? 1 : 0;
		}
		return (double) correct / (double) count;
	}
	
	private boolean close(int x, double y){
		return Math.abs(x - y) <= 0.01; // arbitrary
	}
	
	// Example for estimation.OptimizerIP
	public void runIPExample() throws GRBException{
		estimation.OptimizerIP optimizerIP = new estimation.OptimizerIP(this.trainX, this.trainY, 
				this.train, this.test, this.d, this.sparsity, this.oppositePairs);
		optimizerIP.solve();
		int[] estimate = optimizerIP.classify(this.testX, estimation.OptimizerIP.Rule.Max);
		double accuracy = this.accuracy(estimate, this.testY);
		System.out.println("Accuracy: " + accuracy);
	}
	
	// Example for estimation.OptimizerLP
	public void runLPExample() throws GRBException{
		support.SequentialOptimizer sequentialOptimizer = new support.SequentialOptimizer(this.trainX, this.trainY, 
				this.train, this.d, this.sparsity, this.oppositePairs);
		int[] support = sequentialOptimizer.optimize();
		
		estimation.OptimizerLP optimizerLP = new estimation.OptimizerLP(this.trainX, this.trainY, 
				this.train, this.test, this.d, support);
		optimizerLP.solve();
		int[] estimate = optimizerLP.classify(this.testX, estimation.OptimizerLP.Rule.Max);
		double accuracy = this.accuracy(estimate, this.testY);
		System.out.println("Accuracy: " + accuracy);
	}

	public static void main(String[] args) throws GRBException{
		Example example = new Example();
		example.generateData();
		example.runIPExample();
		example.runLPExample();
	}

}
