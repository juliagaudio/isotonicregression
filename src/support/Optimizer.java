package support;

import java.util.ArrayList;

import gurobi.GRB;
import gurobi.GRBEnv;
import gurobi.GRBException;
import gurobi.GRBLinExpr;
import gurobi.GRBModel;
import gurobi.GRBVar;

public class Optimizer {
	
	private double[][] X;
	private double[] Y;
	private int n;
	private int d;
	private int k;
	private ArrayList<Pair> oppositePairs;
	
	private int[][][] q;
	
	public Optimizer(double[][] X, double[] Y, int n, int d, int k, ArrayList<Pair> oppositePairs) {
		this.X = X;
		this.Y = Y;
		this.n = n;
		this.d = d;
		this.k = k;
		this.oppositePairs = oppositePairs;
		this.findQ();
	}
	
	private int[] lockedCoordinates = null;
	private boolean lockedCoordinatesFlag = false;
	public void setLockedCoordinates(int[] coordinates){
		this.lockedCoordinates = coordinates;
		this.lockedCoordinatesFlag = true;
	}

	
	/**
	 * Encodes coordinate-wise relationships
	 */
	private void findQ(){
		this.q = new int[this.n][this.n][this.d];
		for (int i = 0; i < this.n; i++){
			for (int j = 0; j < this.n; j++){
				for (int p = 0; p < this.d; p++)
				{
					if (X[i][p] - X[j][p] > 0){
						q[i][j][p] = 1;
					}					
				}
			}
		}
			
	}
	
	// VARIABLES
	private GRBVar[] v;
	private void createVVariables() throws GRBException{
		this.v = new GRBVar[this.d];
		for (int i = 0; i < this.d; i++){
			this.v[i] = this.model.addVar(0, 1, 0, GRB.CONTINUOUS, null); 
		}
	}
	
	private GRBVar[][][] c;
	private void createCVariables() throws GRBException{
		this.c = new GRBVar[this.n][this.n][this.d];
		for (int i = 0; i < this.n; i++){
			for (int j = 0; j < this.n; j++){
				for (int k = 0; k < this.d; k++)
				{
					this.c[i][j][k] = this.model.addVar(0, Double.MAX_VALUE, 0, GRB.CONTINUOUS, null);
				}
			}
		}
	}


	
	// CONSTRAINTS
	private void addMonotonicityConstraints() throws GRBException{
		for (int i = 0; i < n; i++){
			for (int j = 0; j < n; j++){
				if (i != j && Y[i] - Y[j] > 0){
					int sum = 0;
					for (int k = 0; k < this.d; k++){
						sum += this.q[i][j][k];
					}
					if (sum > 0){
						GRBLinExpr expr = new GRBLinExpr();
						for (int k = 0; k < this.d; k++){
							expr.addTerm(this.q[i][j][k], this.v[k]);
							expr.addTerm(this.q[i][j][k], this.c[i][j][k]);
						}
						this.model.addConstr(expr, GRB.GREATER_EQUAL, 1, null); 
					}
				}
			}
		}
	}
	
	private void addSparsityConstraint() throws GRBException {
		GRBLinExpr expr = new GRBLinExpr();
		for (int i = 0; i < this.d; i++){
			expr.addTerm(1, this.v[i]);
		}
		this.model.addConstr(expr, GRB.EQUAL, this.k, null);
	}
	
	private void addZeroConstraints() throws GRBException{
		for (int i = 0; i < this.d; i++){
			if (this.lockedCoordinates[i] == 1){
				this.model.addConstr(this.v[i], GRB.EQUAL, 0, null); 
				
				/*The below implementation is somewhat inefficient.
				 * Here we are zeroing out the opposite coordinate.
				 */
				
				for (int j = 0; j < this.oppositePairs.size(); j++){
					Pair pair = this.oppositePairs.get(j);
					int c1 = pair.getC1();
					int c2 = pair.getC2();
					if (i == c1){
						this.model.addConstr(this.v[c2], GRB.EQUAL, 0, null);
						break;
					}
					if (i == c2){
						this.model.addConstr(this.v[c1], GRB.EQUAL, 0, null);
						break;
					}
				}

			}
		}
	}
	
	private void addNoOppositesConstraints() throws GRBException{
		for (int i = 0; i < this.oppositePairs.size(); i++){
			Pair pair = this.oppositePairs.get(i);
			int c1 = pair.getC1();
			int c2 = pair.getC2();
			
			GRBLinExpr expr = new GRBLinExpr();
			expr.addTerm(1, this.v[c1]);
			expr.addTerm(1, this.v[c2]);
			this.model.addConstr(expr, GRB.LESS_EQUAL, 1, null);
		}
	}
	
	
	// OBJECTIVE FUNCTION
	private void addObjectiveFunction() throws GRBException{
		GRBLinExpr obj = new GRBLinExpr();
		for (int i = 0; i < this.n; i++){
			for (int j = 0; j < this.n; j++){
				for (int k = 0; k < this.d; k++){
					obj.addTerm(1, this.c[i][j][k]);
				}
			}
		}
		this.model.setObjective(obj, GRB.MINIMIZE);
	}

	
	private GRBModel model;
	private void initialize() throws GRBException{
		GRBEnv env = new GRBEnv();
		this.model = new GRBModel(env);
		this.model.getEnv().set(GRB.IntParam.OutputFlag, 0);
		this.model.set(GRB.IntAttr.ModelSense, GRB.MINIMIZE);
		this.createVVariables();
		this.createCVariables();
		this.model.update();
		this.addMonotonicityConstraints();
		this.addSparsityConstraint();
		if (this.lockedCoordinatesFlag){
			this.addZeroConstraints();
		}
		this.addNoOppositesConstraints();
		this.addObjectiveFunction();
	}
	
	public void solve() throws GRBException {
		initialize();
		model.update();
		model.optimize();
	}
	
	public int topCoordinate() throws GRBException{
		int coordinate = 0;
		double value = 0;
		for (int i = 0; i < this.d; i++){
			double vValue = this.v[i].get(GRB.DoubleAttr.X);
			if (vValue > value){
				value = vValue;
				coordinate = i;
			}
		}
		return coordinate;
	}
	
	
	public void disposeModel(){
		this.model.dispose();
	}
	
	public int[] coordinates() throws GRBException{
		int[] coordinates = new int[this.d];
		for (int i = 0; i < k; i++){
			int coordinate = -1;
			double value = 0;
			for (int j = 0; j < this.d; j++){
				if (coordinates[j] == 0){
					double vValue = this.v[j].get(GRB.DoubleAttr.X);
					if (vValue > value){
						value = vValue;
						coordinate = j;
					}
				}
			}
			coordinates[coordinate] = 1;
		}
		
		return coordinates;
	}


}

