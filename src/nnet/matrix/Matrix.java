package nnet.matrix;

import java.util.concurrent.ThreadLocalRandom;

import nnet.exception.NNetInvalidMatrixOp;

public class Matrix {
	protected double[][] data;
	protected int rows;
	protected int cols;
	
	/**
	 * Create a matrix with the given rows and columns. Set it to zero.
	 * @param rows
	 * @param cols
	 */
	public Matrix(int rows, int cols)
	{
		this(rows,cols,false);
	}

	/**
	 * Create a matrix with the given rows and columns. 
	 * @param rows
	 * @param cols
	 * @param randomize - true then randomize entries else set to zero.
	 */
	public Matrix(int rows, int cols,boolean randomize)
	{
		this.rows = rows;
		this.cols = cols;
		data = new double[rows][cols];
		if (randomize)
			randomize();
		else
			zero();
	}

	/**
	 * Create a Matrix using an array
	 * @param data
	 */
	public Matrix(double[][] data)
	{
		this.rows = data.length;
		this.cols = data[0].length;
		this.data = data;
	}

	public int getRows() {
		return rows;
	}

	public int getCols() {
		return cols;
	}

	public double[][] getData() {
		return data;
	}
	
	public double get(int row, int col)
	{
		return data[row][col];
	}

	public void set(int row, int col, double v)
	{
		data[row][col] = v;
	}

	public Matrix copy()  {
		Matrix ret = new Matrix(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = data[r][c];		
		return ret;
	}

	public Matrix scale(double m) {
		Matrix ret = new Matrix(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = data[r][c]*m;
		
		return ret;
	}

	public Matrix shift(double b) {
		Matrix ret = new Matrix(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = data[r][c]+b;
		
		return ret;
	}

	public Matrix xlate(double m,double b) {
		Matrix ret = new Matrix(rows,cols);
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret.data[r][c] = (data[r][c]*m)+b;
		
		return ret;
	}

	public Matrix dot(Matrix b) throws NNetInvalidMatrixOp {
		return Matrix.dot(this, b);
	}

	public Matrix add(Matrix b)  {
		return Matrix.add(this, b);
	}

	public Matrix subtract(Matrix b)  {
		return Matrix.subtract(this, b);
	}

	public Matrix transpose()  {
		return Matrix.transpose(this);
	}

	public Matrix square()  {
		return Matrix.squareElement(this);
	}

	public Matrix absoluteValue()  {
		return Matrix.absoluteValue(this);
	}

	public void zero()
	{
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				data[r][c] = 0;			
	}
	
	public double sum()
	{
		double ret = 0;
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				ret += data[r][c];	
		
		return ret;
	}

	public void randomize()
	{
		randomize(-1,1);
	}
	
	public void randomize(double minv, double maxv)
	{
		for (int r=0;r<rows;r++)
			for (int c=0;c<cols;c++)
				data[r][c] = ThreadLocalRandom.current().nextDouble(minv,maxv);		
	}
	
	/**
	 * Return a select row of the matrix
	 * @param r
	 * @return
	 */
	public Matrix getRowVector(int row) {
		Matrix ret = new Matrix(1,cols);
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
	public Matrix getColumnVector(int column) {
		Matrix ret = new Matrix(cols,1);
		for (int r=0;r<rows;r++)
		{
			ret.set(r, 0, data[r][column]);
		}
		
		return ret;
	}

	/*
	 * Return the sum of the given row
	 */
	public double getRowSum(int row) {
		double ret = 0;
		for (int c=0;c<cols;c++)
			ret+= data[row][c];

		return ret;
	}

	/*
	 * Return the sum of the given column
	 */
	public double getColumnSum(int col) {
		double ret = 0;
		for (int r=0;r<rows;r++)
			ret+= data[r][col];

		return ret;
	}
	
	public void print(String name)
	{		
		System.out.println(toString(name));
	}

	public String toString(String name)
	{
		StringBuffer sb = new StringBuffer();
		
		sb.append("Matrix ");
		sb.append(name);
		sb.append(" rows=");
		sb.append(rows);
		sb.append(" cols=");
		sb.append(cols);
		sb.append(" [\n");
		for (int r=0;r<rows;r++)
		{
			sb.append("  {");
			for (int c=0;c<cols;c++)
			{
				double v = data[r][c];
				if (v>=0)
					sb.append(" ");
				sb.append(String.format("%,.8f",v));
				sb.append(" ");				
			}
			
			sb.append("}\n");
		}
		sb.append("]");	
		
		return sb.toString();
	}
	
	/**
	 * Static methods
	 */

	
	/**
	 * Create a new instanced of Martix or a sublcass of Matrix using row and column sizes
	 * @param mclass - the child class
	 * @param nrows - number of rows in new matrix
	 * @param ncols - number of cols in new matric
	 * @return - the new matrix
	 * @throws NNetInvalidMatrixOp 
	 * @throws Exception
	 */
	public static Matrix newInstance(Class<? extends Matrix> mclass,int nrows,int ncols)
	{
		Matrix ret = null;
		
		try {
			ret = mclass.getDeclaredConstructor(int.class,int.class).newInstance(nrows,ncols);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		} 
		
		return ret;
	}

	/**
	 * Static methods
	 * @throws NNetInvalidMatrixOp 
	 */
	public static Matrix dot(Matrix a, Matrix b) throws NNetInvalidMatrixOp
	{
		if (b.rows != a.cols)
			throw new NNetInvalidMatrixOp("Matrix multiply: incompatible arrays");
		
		Matrix ret = newInstance(a.getClass(),a.getRows(),b.getCols());
		
		for (int r=0;r<ret.rows;r++)
			for (int c=0;c<ret.cols;c++)
			{
				for (int s=0; s<a.cols;s++)
					ret.data[r][c] += a.data[r][s] * b.data[s][c];
			}
	
		
		return ret;
	}
	
	public static Matrix add(Matrix a, Matrix b) 
	{
		Matrix ret = newInstance(a.getClass(),a.getRows(),a.getCols());
		
		for (int r=0;r<ret.rows;r++)
			for (int c=0;c<ret.cols;c++)
			{
				ret.data[r][c] += a.data[r][c] + b.data[r][c];
			}
	
		
		return ret;
	}
	
	/**
	 * Subtract b from a
	 * @param a
	 * @param b
	 * @return
	 * @throws NNetInvalidMatrixOp 
	 */
	public static Matrix subtract(Matrix a, Matrix b) 
	{
		Matrix ret = newInstance(a.getClass(),a.getRows(),a.getCols());
		
		for (int r=0;r<ret.rows;r++)
			for (int c=0;c<ret.cols;c++)
			{
				ret.data[r][c] = a.data[r][c] - b.data[r][c];
			}
	
		
		return ret;
	}
	
	/**
	 * return a transpose of the input matrix
	 * @param a 
	 * @return the transposed od input matrix a
	 * @throws NNetInvalidMatrixOp 
	 */
	public static Matrix transpose(Matrix a) 
	{
		Matrix ret = newInstance(a.getClass(),a.getCols(),a.getRows());
		
		for (int r=0;r<a.rows;r++)
			for (int c=0;c<a.cols;c++)
			{
				ret.data[c][r] = a.data[r][c];
			}
	
		
		return ret;
	}
		
	/**
	 * Return a copy of the input matrix with squared values
	 * @param a
	 * @return
	 * @throws NNetInvalidMatrixOp 
	 */
	public static Matrix squareElement(Matrix a) 
	{
		Matrix ret = newInstance(a.getClass(),a.getRows(),a.getCols());
		
		for (int r=0;r<ret.rows;r++)
			for (int c=0;c<ret.cols;c++)
			{
				ret.data[r][c] = a.data[r][c] * a.data[r][c];
			}
	
		
		return ret;
	}

	/**
	 * Return a matrix with the absolute values of the input matrix
	 * @param nNetMatrix
	 * @return
	 */
	public static Matrix absoluteValue(Matrix a) {
		Matrix ret = newInstance(a.getClass(),a.getRows(),a.getCols());
		
		for (int r=0;r<ret.rows;r++)
			for (int c=0;c<ret.cols;c++)
			{
				ret.data[r][c] = Math.abs(a.data[r][c]);
			}
			
		return ret;	
	
	}	

	/**
	 * Create a randomized matrix of the requested size
	 * @throws NNetInvalidMatrixOp 
	 */
	public static Matrix createRandomMatrix(int rows, int cols)  {
		Matrix ret = newInstance(Matrix.class,rows,cols);
		ret.randomize();
		return ret;
	}

	public static Matrix createRandomMatrix(int rows, int cols, double minv, double maxv)  {
		Matrix ret = newInstance(Matrix.class,rows,cols);
		ret.randomize(minv,maxv);
		return ret;
	}

	public static Matrix createRandomMatrix(Class<? extends Matrix> mclass,int rows, int cols) throws NNetInvalidMatrixOp {
		Matrix ret = newInstance(mclass,rows,cols);
		ret.randomize();
		return ret;
	}

	/**
	 * Create an identity matrix (square matrix) 
	 * Rectangular matrices will have the upper left square as ident.
	 * @param rows
	 * @param cols
	 * @return
	 * @throws NNetInvalidMatrixOp 
	 */
	public static Matrix createIdentMatrix(Class<? extends Matrix> mclass,int rows, int cols)  {
		Matrix ret = newInstance(mclass,rows,cols);
		for (int i=0;i<Math.min(rows, cols);i++) {
			ret.data[i][i] = 1; 
		}
		return ret;
	}

	public static Matrix createIdentMatrix(int rows, int cols)  {
		Matrix ret = newInstance(Matrix.class,rows,cols);
		for (int i=0;i<Math.min(rows, cols);i++) {
			ret.data[i][i] = 1; 
		}
		return ret;
	}


}
