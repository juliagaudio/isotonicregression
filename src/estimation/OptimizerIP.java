package estimation;


import java.util.ArrayList;

import support.Pair;
import gurobi.GRB;
import gurobi.GRBEnv;
import gurobi.GRBException;
import gurobi.GRBLinExpr;
import gurobi.GRBModel;
import gurobi.GRBQuadExpr;
import gurobi.GRBVar;

/* This code is tailored for the Noisy Input Model. In particular,
 * the function is binary-valued.
 * */

/**
 * Implements the contrapositive version of the optimization
 * formulation that simultaneously chooses the active coordinates
 * and estimates the function values.
 * @author juliagaudio
 *
 */
public class OptimizerIP {
	
	private double[][] X;
	private double[] Y;
	private int train;
	private int test;
	private int d;
	private int sparsity;
	private ArrayList<Pair> oppositePairs;
		
	public OptimizerIP(double[][] X, double[] Y, int train, int test, int d, int sparsity, ArrayList<Pair> oppositePairs) {
		this.X = X;
		this.Y = Y;
		this.train = train;
		this.test = test;
		this.d = d;
		this.sparsity = sparsity;
		this.oppositePairs = oppositePairs;
	}
	
	private GRBVar[] F;
	private void createFVariables() throws GRBException{
		this.F = new GRBVar[this.train];
		for (int i = 0; i < this.train; i++){
			this.F[i] = this.model.addVar(0, 1, 0, GRB.BINARY, null);
		}
	}
	
	private GRBVar[] z;
	private void createZVariables() throws GRBException{
		this.z = new GRBVar[this.d];
		for (int i = 0; i < this.d; i++){
			this.z[i] = this.model.addVar(0, 1, 0, GRB.BINARY, null);
		}
	}
	
	private void addMonotonicityConstraints() throws GRBException{
		for (int i = 0; i < this.train; i++){
			for (int j = 0; j < this.train; j++){
				GRBLinExpr lhs = new GRBLinExpr();
				for (int k = 0; k < this.d; k++){
					if (this.X[i][k] > this.X[j][k]){ 
						lhs.addTerm(1, this.z[k]);
					}
				}
				GRBLinExpr rhs = new GRBLinExpr();
				rhs.addTerm(1, this.F[i]);
				rhs.addTerm(-1, this.F[j]);
				
				this.model.addConstr(lhs, GRB.GREATER_EQUAL, rhs, null);
			}
		}
	}
	
	private void addSparsityConstraint() throws GRBException{
		GRBLinExpr expr = new GRBLinExpr();
		for (int i = 0; i < this.d; i++){
			expr.addTerm(1, this.z[i]);
		}
		this.model.addConstr(expr, GRB.EQUAL, this.sparsity, null);
	}
	
	// Makes sure that a feature and its negation aren't both included.
	private void addNoOppositesConstraint() throws GRBException{
		for (int i = 0; i < this.oppositePairs.size(); i++){
			Pair pair = this.oppositePairs.get(i);
			int c1 = pair.getC1();
			int c2 = pair.getC2();
			
			GRBLinExpr expr = new GRBLinExpr();
			expr.addTerm(1, this.z[c1]);
			expr.addTerm(1, this.z[c2]);
			this.model.addConstr(expr, GRB.LESS_EQUAL, 1, null);
		}
	}
	
	private void addObjectiveFunction() throws GRBException{
		GRBQuadExpr obj = new GRBQuadExpr();
		for (int i = 0; i < this.train; i++){
			obj.addConstant(this.Y[i]*this.Y[i]);
			obj.addTerm(-2*this.Y[i], this.F[i]);
			obj.addTerm(1, this.F[i], this.F[i]);
		}
		this.model.setObjective(obj);
	}
	
	private GRBModel model;
	private void initialize() throws GRBException{
		GRBEnv env = new GRBEnv();
		this.model = new GRBModel(env);
		this.model.getEnv().set(GRB.IntParam.OutputFlag, 0);
		this.model.set(GRB.IntAttr.ModelSense, GRB.MINIMIZE);
		this.createFVariables();
		this.createZVariables();
		this.model.update();
		this.addMonotonicityConstraints();
		this.addSparsityConstraint();
	    this.addNoOppositesConstraint();
		this.addObjectiveFunction();
	}
	
	private int[] solution;
	public int[] support;
	public void solve() throws GRBException{
		this.initialize();
		this.model.update();
		this.model.optimize();
		
		this.solution = new int[this.train];
		for (int i = 0; i < this.train; i++){
			this.solution[i] = this.round(this.F[i].get(GRB.DoubleAttr.X));
		}
		
		this.support = new int[this.d];
		for (int i = 0; i < this.d; i++){
			this.support[i] = (int) this.z[i].get(GRB.DoubleAttr.X);
		}
		
		this.model.dispose();
	}
	
	private int round(double x){
		return x >= 0.5 ? 1 : 0;
	}
	
	private int classifyMin(double[] x){
		int label = 0;
		for (int j = 0; j < this.train; j++){
			if (this.compare(this.X[j], x, this.support)){ 
				if (this.solution[j] == 1){
					label = 1;
					break;
				}
			}
		}		
		return label;
	}
	
	private int classifyMax(double[] x){				
		int label = 1;
		for (int j = 0; j < this.train; j++){
			if (this.compare(x, this.X[j], this.support)){
				if (this.solution[j] == 0){
					label = 0;
					break;
				}
			}
		}
		
		return label;
	}
	
	private boolean compare(double[] x1, double[] x2, int[] support){
		for (int i = 0; i < this.d; i++){
			if (this.support[i] == 1){
				if (x1[i] > x2[i]){ 
					return false;
				}
			}
		}
		return true;
	}
	
	public enum Rule{Min, Max};
	
	public int[] classify(double[][] testX, Rule rule){
		int[] solution = new int[this.test];
		for (int i = 0; i < this.test; i++){
			if (rule.equals(Rule.Min)){
				solution[i] = this.classifyMin(testX[i]);
			} else {
				solution[i] = this.classifyMax(testX[i]);
			}
		}
		
		return solution;
	}

}
