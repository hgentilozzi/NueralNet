package nnet.matrix.encorders;

import nnet.matrix.NNetMatrix;

public class BaseEncoder implements EncoderIntf {
	private boolean fitted;
	private double[] mean;
	private double[] variance;
	private double[] stdv;

	public BaseEncoder() {
		fitted = false;
		mean = null;
		variance = null;
	}

	private int[] getAllColums(NNetMatrix m) {
		int[] cols = new int[m.getCols()];
		for (int i=0;i<cols.length;i++)
			cols[i] = i;

		return cols;
	}
	/*
	 * Fit data of all columns of the input data
	 */
	@Override
	public void fit(NNetMatrix m) {
		fit(m,getAllColums(m));
	}

	/**
	 * Run a fit operation on the selected columns
	 */
	@Override
	public void fit(NNetMatrix m, int... columns) {
		double[] total = new double[columns.length];;
		
		variance = new double[columns.length];
		mean = new double[columns.length];
		stdv = new double[columns.length];
		
		// Total the columns
		for (int r=0;r<m.getRows();r++)
		{
			int tcol = 0;
			for (int col : columns)
			{
				total[tcol++] += m.get(r, col);
			}
		}

		// Calculate the mean
		for (int mcol=0; mcol< columns.length;mcol++)
		{
			mean[mcol] += total[mcol] / m.getRows();
		}
		
		// Calculate the variance
		double oneOverN = 1.0 / (m.getRows()*1.0);
		for (int r=0;r<m.getRows();r++)
		{
			int vcol = 0;
			for (int col : columns)
			{
				variance[vcol] += (Math.pow((m.get(r, col)-mean[vcol]),2.0)*oneOverN);
				vcol++;
			}
		}
		

		// Calculate the standard deviation
		for (int scol=0; scol< columns.length;scol++)
		{
			stdv[scol] = Math.sqrt(variance[scol]);
		}


		fitted = true;

	}

	@Override
	public NNetMatrix transform(NNetMatrix m) {
		return transform(m,getAllColums(m));
	}

	public NNetMatrix transform(NNetMatrix m,NNetMatrix ret, int... columns) {
		for (int r=0;r<m.getRows();r++)
		{
			int tcol = 0;
			for (int col : columns)
			{
				double z = ((m.get(r,col)-mean[tcol])/stdv[tcol]);
				ret.set(r,col,z);
				tcol++;
			}
		}
		
		return ret;
	}

	@Override
	public NNetMatrix transform(NNetMatrix m, int... columns) {
		return transform(m,new NNetMatrix(m.getRows(),m.getRows()),columns);
	}

	@Override
	public NNetMatrix transformInPlace(NNetMatrix m) {
		return transform(m,m,getAllColums(m));
	}

	@Override
	public NNetMatrix transformInPlace(NNetMatrix m, int... columns) {
		return transform(m,m,columns);
	}

	@Override
	public NNetMatrix fitAndTransform(NNetMatrix m) {
		fit(m);
		return transform(m);
	}

	@Override
	public NNetMatrix fitAndTransform(NNetMatrix m, int... columns) {
		fit(m);
		return transform(m,columns);
	}

	@Override
	public NNetMatrix fitAndTransformInPlace(NNetMatrix m) {
		fit(m);
		return transform(m,m);
	}

	@Override
	public NNetMatrix fitAndTransformInPlace(NNetMatrix m, int... columns) {
		fit(m);
		return transform(m,m,columns);
	}

	@Override
	public double[] getMean() {
		return mean;
	}

	@Override
	public double[] getVariance() {
		return variance;
	}

	@Override
	public boolean isFitted() {
		return fitted;
	}

	@Override
	public double[] getStdv() {
		return stdv;
	}

}
