package estimation;

import gurobi.GRB;
import gurobi.GRBEnv;
import gurobi.GRBException;
import gurobi.GRBLinExpr;
import gurobi.GRBModel;
import gurobi.GRBVar;

/* This code is tailored for the Noisy Input Model. In particular,
 * the function is binary-valued.
 * */
public class OptimizerLP {

	private double[][] X;
	private double[] Y;
	private int train;
	private int test;
	private int d;
	int[] support;

	public OptimizerLP(double[][] X, double[] Y, int train, int test, int d, int[] support) {
		this.X = X;
		this.Y = Y;
		this.train = train;
		this.test = test;
		this.d = d;
		this.support = support;
	}

	// VARIABLES	
	private GRBVar[] F;
	private void createFVariables() throws GRBException{
		this.F = new GRBVar[this.train];
		for (int i = 0; i < this.train; i++){
			this.F[i] = this.model.addVar(0, 1, 0, GRB.BINARY, null); 
		}
	}

	// CONSTRAINTS
	private void addMonotonicityConstraints() throws GRBException{
		for (int i = 0; i < this.train; i++){
			for (int j = 0; j < this.train; j++){
				GRBLinExpr lhs = new GRBLinExpr();
				for (int k = 0; k < this.d; k++){
					if (this.X[i][k] > this.X[j][k]){
						lhs.addConstant(this.support[k]);
					}
				}
				GRBLinExpr rhs = new GRBLinExpr();
				rhs.addTerm(1, this.F[i]);
				rhs.addTerm(-1, this.F[j]);

				this.model.addConstr(lhs, GRB.GREATER_EQUAL, rhs, null);
			}
		}
	}

	private boolean compare(double[] x1, double[] x2){
		for (int i = 0; i < this.d; i++){
			if (this.support[i] == 1){
				if (x1[i] > x2[i]){ 
					return false;
				}
			}
		}
		return true;
	}

	private void addObjectiveFunction() throws GRBException{
		// In general a quadratic objective is required, but for binary-valued functions things simplify.
		GRBLinExpr obj = new GRBLinExpr();
		for (int i = 0; i < this.train; i++){
			// The following is tailored for binary-valued input. 
			if (this.Y[i] >= 0.5){
				obj.addConstant(1);
				obj.addTerm(-1, this.F[i]);
			} else {
				obj.addTerm(1, this.F[i]);
			}
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
		this.model.update();
		this.addMonotonicityConstraints();
		this.addObjectiveFunction();
	}

	private int[] solution;
	public void solve() throws GRBException{
		this.initialize();
		this.model.update();
		this.model.optimize();

		this.solution = new int[this.train];
		for (int i = 0; i < this.train; i++){
			this.solution[i] = this.round(this.F[i].get(GRB.DoubleAttr.X));
		}		
		this.model.dispose(); 
	}

	private int round(double x){
		return x >= 0.5 ? 1 : 0;
	}

	private int classifyMin(double[] x){
		int label = 0;
		for (int j = 0; j < this.train; j++){
			if (this.compare(this.X[j], x)){
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
			if (this.compare(x, this.X[j])){
				if (this.solution[j] == 0){
					label = 0;
					break;
				}
			}
		}		
		return label;
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
