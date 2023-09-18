package nnet.test.matrix;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import nnet.exception.NNetInvalidMatrixOp;
import nnet.matrix.Matrix;


class TestMatrix {

	@Test
	void testMultuplySqrMatic() {
		Matrix x = new Matrix(2,2);
		Matrix y = new Matrix(2,2);
		
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		y.set(0, 0, 5);
		y.set(0, 1, 6);
		y.set(1, 0, 7);
		y.set(1, 1, 8);
		
		Matrix z=null;
		try {
			z = Matrix.dot(x, y);
		} catch (NNetInvalidMatrixOp e) {
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
		Matrix x = new Matrix(3,2);
		Matrix y = new Matrix(2,3);
		
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
		
		Matrix z=null;
		try {
			z = Matrix.dot(x, y);
		} catch (NNetInvalidMatrixOp e) {
			e.printStackTrace();
		}

		assertEquals(3,z.getRows());
		assertEquals(3,z.getCols());

		
	}

	@Test
	void testTransPose() {
		Matrix x = new Matrix(2,2);
		
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		Matrix z = x.transpose();
		
		assertEquals(1.0,z.get(0,0));
		assertEquals(3.0,z.get(0,1));
		assertEquals(2.0,z.get(1,0));
		assertEquals(4.0,z.get(1,1));
	}

	@Test
	void testAdd() {
		Matrix x = new Matrix(2,2);
		Matrix y = new Matrix(2,2);
		
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		y.set(0, 0, 5);
		y.set(0, 1, 6);
		y.set(1, 0, 7);
		y.set(1, 1, 8);
		
		Matrix z = Matrix.add(x, y);
		
		assertEquals(6.0,z.get(0,0));
		assertEquals(8.0,z.get(0,1));
		assertEquals(10.0,z.get(1,0));
		assertEquals(12.0,z.get(1,1));
	}

	@Test
	void testSubtract() {
		Matrix x = new Matrix(2,2);
		Matrix y = new Matrix(2,2);
		
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		y.set(0, 0, 5);
		y.set(0, 1, 6);
		y.set(1, 0, 7);
		y.set(1, 1, 8);
		
		Matrix z = Matrix.subtract(x, y);
		
		assertEquals(-4.0,z.get(0,0));
		assertEquals(-4.0,z.get(0,1));
		assertEquals(-4.0,z.get(1,0));
		assertEquals(-4.0,z.get(1,1));
	}
	
	@Test
	void testFromArray() {
		double[][] data = new double[][] {{1,2,3},{3,4,5}};
		Matrix x = new Matrix(data);
		
		assertEquals(2,x.getRows());
		assertEquals(3,x.getCols());
		
	}
		
	@Test
	void testgetRowVector() {
		// Square matrix
		Matrix x = Matrix.createIdentMatrix(2,2);
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		Matrix rowvec = x.getRowVector(0);
		assertEquals(1, rowvec.get(0, 0));
		assertEquals(2, rowvec.get(0, 1));

		rowvec = x.getRowVector(1);
		assertEquals(3, rowvec.get(0, 0));
		assertEquals(4, rowvec.get(0, 1));
		
	}
	
	@Test
	void testgetColVector() {
		// Square matrix
		Matrix x = Matrix.createIdentMatrix(2,2);
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
		
		Matrix colvec = x.getColumnVector(0);
		assertEquals(1, colvec.get(0, 0));
		assertEquals(3, colvec.get(1, 0));
	
		colvec = x.getColumnVector(1);
		assertEquals(2, colvec.get(0, 0));
		assertEquals(4, colvec.get(1, 0));
		
	}
	
	@Test
	void testSquare() {
		Matrix x = new Matrix(2,2);
		x.set(0, 0, 1);
		x.set(0, 1, 2);
		x.set(1, 0, 3);
		x.set(1, 1, 4);
	
		x = x.square();
		
		assertEquals(1,x.get(0,0));
		assertEquals(4,x.get(0,1));
		assertEquals(9,x.get(1,0));
		assertEquals(16,x.get(1,1));

		
	}
	
	@Test
	void testCreateIdentMatrix() {
		// Square matrix
		Matrix x = Matrix.createIdentMatrix(2,2);
	
		for (int r=0;r<x.getRows();r++) {
			for (int c=0;c<x.getCols();c++) {
				if (r==c)
					assertEquals(1, x.get(r, c));
				else
					assertEquals(0, x.get(r, c));
			}
		}
	
		// Horizontal rectangle matrix
		x = Matrix.createIdentMatrix(2,3);
	
		for (int r=0;r<x.getRows();r++) {
			for (int c=0;c<x.getCols();c++) {
				if (r==c)
					assertEquals(1, x.get(r, c));
				else
					assertEquals(0, x.get(r, c));
			}
		}
		
		// Vertical rectangle matrix
		x = Matrix.createIdentMatrix(5,2);
	
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
