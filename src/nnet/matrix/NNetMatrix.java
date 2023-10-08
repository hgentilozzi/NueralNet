package nnet.matrix;

import java.util.function.Function;

import mxlib.excep.MxlibInvalidMatrixOp;
import mxlib.matrix.Matrix;
import mxlib.matrix.MxArray;
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


	/**
	 * Generic update of matrix
	 * @param m - scalar
	 * @return new vector as (this * m)
	 */
	public NNetMatrix matrixFunc(Function<Double,Double> func) {
		NNetMatrix ret = new NNetMatrix(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = func.apply(data[r][c]);
		
		return ret;
	}
	public NNetMatrix copyAndRelu()  {
		return matrixFunc(v -> (v>=0? v : 0));
	}

	
	public NNetMatrix getSoftMax() {
		NNetMatrix ret = new NNetMatrix(rows,cols);
		for (int r=0;r<rows;r++)
		{
			double rowSum = getRowSum(r);
			if (rowSum==0.0)
				rowSum = 1.0;
			
			for (int c=0;c<cols;c++)
				ret.data[r][c] = data[r][c] / rowSum;
		}
		
		return ret;
	}

	public NNetMatrix scale(double m) {
		return matrixFunc(d -> d*m);
	}

	public NNetMatrix shift(double b) {
		return matrixFunc(d -> d+b);
	}

	public NNetMatrix xlate(double m,double b) {
		NNetMatrix ret = new NNetMatrix(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = (data[r][c]*m)+b;
		
		return ret;
	}

	public NNetMatrix dot(NNetMatrix b) throws MxlibInvalidMatrixOp {
		return new NNetMatrix(MxArray.dot(this.data, b.data));
	}

	public NNetMatrix add(NNetMatrix b) throws MxlibInvalidMatrixOp  {
		return new NNetMatrix(MxArray.add(this.data, b.data));
	}

	public NNetMatrix subtract(NNetMatrix b) throws MxlibInvalidMatrixOp  {
		return new NNetMatrix(MxArray.subtract(this.data, b.data));
	}

	public NNetMatrix transpose() {
		return new NNetMatrix(MxArray.transpose(this.data));
	}

	public NNetMatrix squareElement() {
		return matrixFunc(d -> d*d);
	}
	
	public Matrix absoluteValue() {
		return matrixFunc(d -> Math.abs(d));
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
		return new NNetMatrix(rows,cols,true);
	}

	public static NNetMatrix createRandomMatrix(int rows, int cols, double minv, double maxv)  {
		NNetMatrix ret = new NNetMatrix(rows,cols);
		ret.randomize(minv,maxv);
		return ret;
	}

	public static NNetMatrix createRandomMatrix(int rows, int cols, double fillValue)  {
		NNetMatrix ret = new NNetMatrix(rows,cols);
		ret.fill(fillValue);
		return ret;
	}

	/**
	 * Add a bias to the input matrix
	 * @param bias - one row vector. Apply the same bias to all rows of this matrix
	 * @return
	 */
	public NNetMatrix bias(NNetMatrix bias) {
		
		if (bias==null)
			return this;
		
		NNetMatrix ret = new NNetMatrix(rows,cols);

		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = data[r][c] + bias.data[0][c];
		
		return ret;
	}

	
	/**
	 * Static methods
	 */
	public static NNetMatrix createIdentMatrix(int rows) {
		return new NNetMatrix(MxArray.createIdent(rows));
	}

	


}
