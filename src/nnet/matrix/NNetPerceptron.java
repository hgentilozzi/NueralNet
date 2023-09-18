package nnet.matrix;

import nnet.exception.NNetInvalidMatrixOp;
import nnet.matrix.acvt.ActivationFunction;

public class NNetPerceptron extends Matrix {

	public NNetPerceptron(double[][] data) {
		super(data);
	}

	public NNetPerceptron(int rows, int cols, boolean randomize) {
		super(rows, cols, randomize);
	}

	public NNetPerceptron(int rows, int cols) {
		super(rows, cols);
	}

	public NNetPerceptron copy()  {
		NNetPerceptron ret = new NNetPerceptron(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = data[r][c];		
		return ret;
	}


	public NNetPerceptron copyAndRelu()  {
		NNetPerceptron ret = new NNetPerceptron(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = (data[r][c]>=0)? data[r][c] : 0;		
		return ret;
	}
	
	public NNetPerceptron getActivation(ActivationFunction af) {
		NNetPerceptron ret = new NNetPerceptron(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = af.f(data[r][c]);
		
		return ret;
	}

	
	public NNetPerceptron getGradient(ActivationFunction af) {
		if (af==null)
			return copy();
		
		NNetPerceptron ret = new NNetPerceptron(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = af.grad(data[r][c]);
		
		return ret;
	}

	public NNetPerceptron scale(double scaler) {
		NNetPerceptron ret = new NNetPerceptron(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = data[r][c]*scaler;
		
		return ret;
	}

	public NNetPerceptron dot(NNetPerceptron b) throws NNetInvalidMatrixOp {
		return (NNetPerceptron) Matrix.dot(this, b);
	}

	public NNetPerceptron add(NNetPerceptron b)  {
		return (NNetPerceptron) Matrix.add(this, b);
	}

	public NNetPerceptron subtract(NNetPerceptron b)  {
		return (NNetPerceptron) Matrix.subtract(this, b);
	}

	public NNetPerceptron transpose() {
		return (NNetPerceptron) Matrix.transpose(this);
	}

	public NNetPerceptron squareElement() {
		return (NNetPerceptron) Matrix.squareElement(this);
	}
	
	public static NNetPerceptron createIdentMatrix(int rows, int cols)  {
		return (NNetPerceptron) Matrix.createIdentMatrix(NNetPerceptron.class,rows,cols);
	}
	
	/**
	 * Return a select row of the matrix
	 * @param r
	 * @return
	 */
	public NNetPerceptron getRowVector(int row) {
		NNetPerceptron ret = new NNetPerceptron(1,cols);
		for (int c=0;c<cols;c++)
		{
			ret.set(0, c, data[row][c]);
		}
		
		return ret;
	}

	/**
	 * Return a select row of the matrix
	 * @param r
	 * @return
	 */
	public NNetPerceptron getColumnVector(int column) {
		NNetPerceptron ret = new NNetPerceptron(cols,1);
		for (int r=0;r<rows;r++)
		{
			ret.set(r, 0, data[r][column]);
		}
		
		return ret;
	}
	
	public static NNetPerceptron createRandomMatrix(int rows, int cols)  {
		NNetPerceptron ret = (NNetPerceptron) newInstance(NNetPerceptron.class,rows,cols);
		ret.randomize();
		return ret;
	}
	


}
