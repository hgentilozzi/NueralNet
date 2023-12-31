package nnet.test.matrix;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import mxlib.excep.MxlibInvalidMatrixOp;
import nnet.matrix.NNetMatrix;

class TestNNetMatrix {
	
	@Test
	void testScale() {
		NNetMatrix x = new NNetMatrix(2,2);
		
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		NNetMatrix z=null;
		z = x.scale(3);
		
		assertEquals(3,z.get(0,0));
		assertEquals(6,z.get(0,1));
		assertEquals(9,z.get(1,0));
		assertEquals(12,z.get(1,1));
		
	}

	@Test
	void testShift() {
		NNetMatrix x = new NNetMatrix(2,2);
		
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		NNetMatrix z=null;
		z = x.shift(3);
		
		assertEquals(4,z.get(0,0));
		assertEquals(5,z.get(0,1));
		assertEquals(6,z.get(1,0));
		assertEquals(7,z.get(1,1));
		
	}
	
	@Test
	void testXlate() {
		NNetMatrix x = new NNetMatrix(2,2);
		
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		NNetMatrix z=null;
		z = x.xlate(3,5);
		
		assertEquals(8,z.get(0,0));
		assertEquals(11,z.get(0,1));
		assertEquals(14,z.get(1,0));
		assertEquals(17,z.get(1,1));
		
	}


	@Test
	void testMultuplySqrMatic() {
		NNetMatrix x = new NNetMatrix(2,2);
		NNetMatrix y = new NNetMatrix(2,2);
		
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		y.set(0, 0, 5);
		y.set(0, 1, 6);
		y.set(1, 0, 7);
		y.set(1, 1, 8);
		
		NNetMatrix z=null;
		try {
			z = x.dot(y);
		} catch (MxlibInvalidMatrixOp e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		assertEquals(19.0,z.get(0,0));
		assertEquals(22.0,z.get(0,1));
		assertEquals(43.0,z.get(1,0));
		assertEquals(50.0,z.get(1,1));
		
	}
	
	@Test
	void testMultuplyRectMatic() {
		NNetMatrix x = new NNetMatrix(3,2);
		NNetMatrix y = new NNetMatrix(2,3);
		
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		x.set(2, 0, 5);
		x.set(2, 1, 6);
		
		y.set(0, 0, 5);
		y.set(0, 1, 6);
		y.set(0, 2, 9);
		y.set(1, 0, 7);
		y.set(1, 1, 8);
		y.set(1, 2, 10);
		
		NNetMatrix z=null;
		try {
			z = x.dot(y);
		} catch (MxlibInvalidMatrixOp e) {
			e.printStackTrace();
		}

		assertEquals(3,z.getRows());
		assertEquals(3,z.getCols());

		
	}

	@Test
	void testTransPose() {
		NNetMatrix x = new NNetMatrix(2,2);
		
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		NNetMatrix z = x.transpose();
		
		assertEquals(1.0,z.get(0,0));
		assertEquals(3.0,z.get(0,1));
		assertEquals(2.0,z.get(1,0));
		assertEquals(4.0,z.get(1,1));
	}

	@Test
	void testAdd() throws MxlibInvalidMatrixOp {
		NNetMatrix x = new NNetMatrix(2,2);
		NNetMatrix y = new NNetMatrix(2,2);
		
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		y.set(0, 0, 5);
		y.set(0, 1, 6);
		y.set(1, 0, 7);
		y.set(1, 1, 8);
		
		NNetMatrix z = x.add(y);
		
		assertEquals(6.0,z.get(0,0));
		assertEquals(8.0,z.get(0,1));
		assertEquals(10.0,z.get(1,0));
		assertEquals(12.0,z.get(1,1));
	}

	@Test
	void testSubtract() throws MxlibInvalidMatrixOp {
		NNetMatrix x = new NNetMatrix(2,2);
		NNetMatrix y = new NNetMatrix(2,2);
		
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		y.set(0, 0, 5);
		y.set(0, 1, 6);
		y.set(1, 0, 7);
		y.set(1, 1, 8);
		
		NNetMatrix z = x.subtract(y);
		
		assertEquals(-4.0,z.get(0,0));
		assertEquals(-4.0,z.get(0,1));
		assertEquals(-4.0,z.get(1,0));
		assertEquals(-4.0,z.get(1,1));
	}
	
	@Test
	void testFromArray() {
		double[][] data = new double[][] {{1,2,3},{3,4,5}};
		NNetMatrix x = new NNetMatrix(data);
		
		assertEquals(2,x.getRows());
		assertEquals(3,x.getCols());
		
	}
		
	@Test
	void testgetRowVector() {
		// Square matrix
		NNetMatrix x = (NNetMatrix) NNetMatrix.createIdentMatrix(2);
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		NNetMatrix rowvec = x.getRowVector(0);
		assertEquals(1, rowvec.get(0, 0));
		assertEquals(2, rowvec.get(0, 1));

		rowvec = x.getRowVector(1);
		assertEquals(3, rowvec.get(0, 0));
		assertEquals(4, rowvec.get(0, 1));
		
	}
	
	@Test
	void testgetColVector() {
		// Square matrix
		NNetMatrix x = (NNetMatrix) NNetMatrix.createIdentMatrix(2);
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		NNetMatrix colvec = x.getColumnVector(0);
		assertEquals(1, colvec.get(0, 0));
		assertEquals(3, colvec.get(1, 0));
	
		colvec = x.getColumnVector(1);
		assertEquals(2, colvec.get(0, 0));
		assertEquals(4, colvec.get(1, 0));
		
	}
	
	@Test
	void testSquare() {
		NNetMatrix x = new NNetMatrix(2,2);
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
	
		x = x.squareElement();
		
		assertEquals(1,x.get(0,0));
		assertEquals(4,x.get(0,1));
		assertEquals(9,x.get(1,0));
		assertEquals(16,x.get(1,1));	
	}
	
	@Test
	void testSoftMax() {
		NNetMatrix x = new NNetMatrix(2,2);
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
	
		x = x.getSoftMax();
		
		assertEquals(1.0/3.0,x.get(0,0));
		assertEquals(2.0/3.0,x.get(0,1));
		assertEquals(3.0/7.0,x.get(1,0));
		assertEquals(4.0/7.0,x.get(1,1));	
	}
		
	@Test
	void testCreateIdentMatrix() {
		// Square matrix
		NNetMatrix x = NNetMatrix.createIdentMatrix(2);
	
		for (int r=0;r<x.getRows();r++) {
			for (int c=0;c<x.getCols();c++) {
				if (r==c)
					assertEquals(1, x.get(r, c));
				else
					assertEquals(0, x.get(r, c));
			}
		}
	}

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

}
