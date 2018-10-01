package net.hudup.temp;

import java.util.List;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.NextUpdate;
import net.hudup.regression.AbstractRM;
import net.hudup.regression.VarWrapper;
import net.hudup.regression.em.ui.graph.Graph;

/**
 * This class implements regression model with correlation in case of missing data, called COR algorithm.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@NextUpdate
public class CorRegression extends AbstractRM implements DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of X vectors
	 */
	protected List<List<Double>> xVectors = Util.newList();
	
	
	/**
	 * Z vector.
	 */
	protected List<Double> zVector = Util.newList();

	
	/**
	 * Default constructor.
	 */
	public CorRegression() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	
	@Override
	public Object learn0() throws Exception {
		int n = xIndices.size();
		List<double[]> A = Util.newList(n);
		List<Double> b = Util.newList(n);
		
		for (int i = 0; i < n; i++) {
			double covXiZ = cor(xVectors.get(i), zVector);
			b.add(covXiZ);
			
			double[] vector = new double[n];
			A.add(vector);
		}
		
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				double covXiXj = cor(xVectors.get(i), xVectors.get(j));
				A.get(i)[j] = covXiXj;
			}
		}

		//Due to correlation matrix is symmetric
		for (int i = n-1; i >= 1; i--) {
			for (int j = i-1; j >= 0; j--) {
				A.get(i)[j] = A.get(j)[i];
			}
		}
		
		this.coeffs = AbstractRM.solve(A, b);
		if (this.coeffs == null)
			return null;
		
		//Adjusting intercept, improved later
		double sumZ = 0;
		int nZ = 0;
		for (int i = 0; i < zVector.size(); i++) {
			double z = zVector.get(i) != null ? zVector.get(i) : Constants.UNUSED; 
			if (Util.isUsed(z)) {
				sumZ += zVector.get(i);
				nZ++;
			}
		}
		this.coeffs.set(0, this.coeffs.get(0) + sumZ / (double)nZ);
		//Adjusting intercept, improved later
		
		return this.coeffs;
	}

	
	/**
	 * Calculating the correlation between two vectors. This method will be re-implemented in future.
	 * @param xVector specified vector x.
	 * @param yVector specified vector y.
	 * @return the correlation between two vectors.
	 */
	private double cor(List<Double> xVector, List<Double> yVector) {
		if (xVector.size() == 0 || yVector.size() == 0)
			return 0;
		
		int N = Math.min(xVector.size(), yVector.size());
		List<Integer> U = Util.newList(N);
		double meanX = 0, meanY = 0;
		for (int i = 0; i < N; i++) {
			double x = xVector.get(i) != null ? xVector.get(i) : Constants.UNUSED; 
			double y = yVector.get(i) != null ? yVector.get(i) : Constants.UNUSED; 
			if (!Util.isUsed(x) || !Util.isUsed(y))
				continue;
			
			meanX += x;
			meanY += y;
			U.add(i);
		}
		if (U.size() == 0)
			return 0;
		
		meanX = meanX / (double)U.size();
		meanY = meanY / (double)U.size();
		double cov = 0;
		double varX = 0;
		double varY = 0;
		for (int i = 0; i < U.size(); i++) {
			int k = U.get(i);
			double x = xVector.get(k) - meanX;
			double y = yVector.get(k) - meanY;
			cov += x * y;
			varX += x * x;
			varY += y * y;
		}

		if (varX != 0 && varY != 0) {
			return cov / (Math.sqrt(varX*varY));
		}
		else {
			if (varX == 0 && varY == 0)
				return 1;
			else
				return 0;
		}
	}

	
	@Override
	protected boolean prepareInternalData() throws Exception {
		// TODO Auto-generated method stub
		if (!super.prepareInternalData())
			return false;
		
		//Begin extracting data
		while (this.sample.next()) {
			Profile profile = this.sample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			List<Double> xVector0 = null;
			if (this.xVectors.size() <= 0) {
				xVector0 = Util.newList();
				this.xVectors.add(xVector0);
			}
			xVector0 = this.xVectors.get(0);
			xVector0.add(1.0);
			
			for (int j = 1; j < this.xIndices.size(); j++) {
				List<Double> xVector = null;
				if (this.xVectors.size() <= j) {
					xVector = Util.newList();
					this.xVectors.add(xVector);
				}
				xVector = this.xVectors.get(j);

				double value = extractRegressorValue(profile, j);
				xVector.add((double)transformRegressor(value, false));
			}
			
			double lastValue = extractNumber(extractResponseValue(profile));
			this.zVector.add((double)transformResponse(lastValue, false));
		}
		this.sample.reset();
		//End extracting data
		
		return true;
	}

	
	@Override
	public synchronized void unsetup() {
		super.unsetup();
		xVectors.clear();
		zVector.clear();
	}

	
	@Override
	protected void clearInternalData() {
		// TODO Auto-generated method stub
		super.clearInternalData();
		xVectors.clear();
		zVector.clear();
	}


	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "corr";
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		CorRegression cor = new CorRegression();
		cor.getConfig().putAll((DataConfig)this.getConfig().clone());
		return cor;
	}


	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}


	@Override
	public List<VarWrapper> extractRegressors() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public List<VarWrapper> extractSingleRegressors() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public Object executeByXStatistic(double[] xStatistic) {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public Graph createRegressorGraph(int xIndex) {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public Graph createResponseGraph() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public Graph createErrorGraph() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public List<Graph> createResponseRalatedGraphs() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public double calcVariance() {
		// TODO Auto-generated method stub
		return 0;
	}


	@Override
	public double calcR() {
		// TODO Auto-generated method stub
		return 0;
	}


	@Override
	public double[] calcError() {
		// TODO Auto-generated method stub
		return null;
	}
	
	
}
