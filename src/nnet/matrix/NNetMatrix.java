package nnet.matrix;

import nnet.exception.NNetInvalidMatrixOp;
import nnet.matrix.acvt.ActivationFunction;

public class NNetMatrix extends Matrix {

	public NNetMatrix(double[][] data) {
		super(data);
	}

	public NNetMatrix(int rows, int cols, boolean randomize) {
		super(rows, cols, randomize);
	}

	public NNetMatrix(int rows, int cols) {
		super(rows, cols);
	}

	public NNetMatrix copy()  {
		NNetMatrix ret = new NNetMatrix(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = data[r][c];		
		return ret;
	}


	public NNetMatrix copyAndRelu()  {
		NNetMatrix ret = new NNetMatrix(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = (data[r][c]>=0)? data[r][c] : 0;		
		return ret;
	}
	
	public NNetMatrix getActivation(ActivationFunction af) {
		NNetMatrix ret = new NNetMatrix(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
			{
				ret.data[r][c] = af.f(data[r][c]);
			}
		
		return ret;
	}

	
	public NNetMatrix getGradient(ActivationFunction af) {
		if (af==null)
			return copy();
		
		NNetMatrix ret = new NNetMatrix(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = af.grad(data[r][c]);
		
		return ret;
	}

	public NNetMatrix scale(double scaler) {
		NNetMatrix ret = new NNetMatrix(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = data[r][c]*scaler;
		
		return ret;
	}

	public NNetMatrix shift(double bias) {
		NNetMatrix ret = new NNetMatrix(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = data[r][c]+bias;
		
		return ret;
	}

	public NNetMatrix xlate(double m,double b) {
		NNetMatrix ret = new NNetMatrix(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = (data[r][c]*m)+b;
		
		return ret;
	}

	public NNetMatrix dot(NNetMatrix b) throws NNetInvalidMatrixOp {
		return (NNetMatrix) Matrix.dot(this, b);
	}

	public NNetMatrix add(NNetMatrix b)  {
		return (NNetMatrix) Matrix.add(this, b);
	}

	public NNetMatrix subtract(NNetMatrix b)  {
		return (NNetMatrix) Matrix.subtract(this, b);
	}

	public NNetMatrix transpose() {
		return (NNetMatrix) Matrix.transpose(this);
	}

	public NNetMatrix squareElement() {
		return (NNetMatrix) Matrix.squareElement(this);
	}
	
	public static NNetMatrix createIdentMatrix(int rows, int cols)  {
		return (NNetMatrix) Matrix.createIdentMatrix(NNetMatrix.class,rows,cols);
	}
	
	/**
	 * Return a select row of the matrix
	 * @param r
	 * @return
	 */
	public NNetMatrix getRowVector(int row) {
		NNetMatrix ret = new NNetMatrix(1,cols);
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
	public NNetMatrix getColumnVector(int column) {
		NNetMatrix ret = new NNetMatrix(cols,1);
		for (int r=0;r<rows;r++)
		{
			ret.set(r, 0, data[r][column]);
		}
		
		return ret;
	}

	
	public static NNetMatrix createRandomMatrix(int rows, int cols)  {
		NNetMatrix ret = (NNetMatrix) newInstance(NNetMatrix.class,rows,cols);
		ret.randomize();
		return ret;
	}

	public static NNetMatrix createRandomMatrix(int rows, int cols, double minv, double maxv)  {
		NNetMatrix ret = (NNetMatrix) newInstance(NNetMatrix.class,rows,cols);
		ret.randomize(minv,maxv);
		return ret;
	}
	


}
