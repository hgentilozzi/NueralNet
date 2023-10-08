package nnet.matrix.acvt;

import nnet.matrix.NNetMatrix;

public class ActivationFunctionSoftMax extends ActivationFunctionLinear {

	public ActivationFunctionSoftMax() {
	}

	@Override
	public NNetMatrix f(NNetMatrix m) {
		NNetMatrix ret = new NNetMatrix(m.getRows(),m.getCols());
		for (int r=0;r<m.getRows();r++)
		{
			double rowSum = getRowSum(m,r);
			
			for (int c=0;c<m.getCols();c++)
				ret.set(r,c,szi(m.get(r, c),rowSum ));
		}
		
		return ret;	}

	@Override
	public NNetMatrix grad(NNetMatrix m) {
		NNetMatrix ret = new NNetMatrix(m.getRows(),m.getCols());

		for (int r=0;r<m.getRows();r++)
		{
			double rowSum = getRowSum(m,r);

			for (int c=0;c<m.getCols();c++)
			{
				double dv = 0;
				for (int s=0;s<m.getCols();s++)
				{
					if (c==s)
						dv += szi(m.get(r, s),rowSum ) * (1 - szi(m.get(r, s),rowSum ));
					else 
						dv += szi(m.get(r, s),rowSum ) * szi(m.get(r, c),rowSum );						
				}
				
				ret.set(r,c,dv);
			}
		}

		
		return ret;
	}

	private double szi(double z, double sum) {
		return Math.exp(z) / sum;
	}

	/*
	 * Return the sum of the given row
	 */
	public double getRowSum(NNetMatrix m,int row) {
		double ret = 0;
		for (int c=0;c<m.getCols();c++)
			ret+= Math.exp(m.get(row,c));

		return ret;
	}

}
