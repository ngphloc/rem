package net.hudup.regression.em;

import java.util.Arrays;
import java.util.List;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.data.DataConfig;
import net.hudup.core.logistic.DSUtil;

/**
 * This class implements default expectation maximization algorithm for regression model in case of missing data, called REM algorithm.
 * Moreover, the prior probability of parameter is used to improve the estimation.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RegressionEMPrior extends RegressionEMImpl {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of mean of prior distribution.
	 */
	public static final String MEAN0_FIELD = "mean0";

			
	/**
	 * Default mean of prior distribution.
	 */
	public static final double MEAN0_DEFAULT = Constants.UNUSED;

	
	/**
	 * Name of variance of prior distribution.
	 */
	public static final String VARIANCE0_FIELD = "variance0";

			
	/**
	 * Default variance of prior distribution.
	 */
	public static final double VARIANCE0_DEFAULT = Constants.UNUSED;

	
	/**
	 * Prior mean.
	 */
	protected double mean0 = 0;
	
	
	/**
	 * Prior variance.
	 */
	protected double variance0 = 0;

	
	/**
	 * Default constructor.
	 */
	public RegressionEMPrior() {
		// TODO Auto-generated constructor stub
		super();
	}
	
	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "prior_rem";
	}

	
	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		RegressionEMPrior em = new RegressionEMPrior();
		em.getConfig().putAll((DataConfig)this.getConfig().clone());
		return em;
	}


	@Override
	protected Object maximization(Object currentStatistic, Object...info) throws Exception {
		// TODO Auto-generated method stub
		LargeStatistics stat = (LargeStatistics)currentStatistic;
		if (stat.isEmpty())
			return null;
		List<double[]> zStatistic = stat.getZData();
		List<double[]> xStatistic = stat.getXData();
		int N = zStatistic.size();
		ExchangedParameter currentParameter = (ExchangedParameter)getCurrentParameter();
		
		//Begin calculating prior parameter. It is not tested in current version yet.
		double zFactor = Constants.UNUSED;
		List<double[]> zDataToLearnAlpha = Util.newList(zStatistic.size());
		if (currentParameter != null) {
			double variance = currentParameter.estimateZVariance(stat);
			zFactor = this.variance0 - variance;
			
			for (int i = 0; i < N; i++) {
				double[] zVector = Arrays.copyOf(zStatistic.get(i), zStatistic.get(i).length); 
				zVector[1] = this.variance0*zVector[1] - variance*this.mean0;
				
				zDataToLearnAlpha.add(zVector);
			}
		}
		else
			zDataToLearnAlpha = zStatistic; 
		//End calculating prior parameter
		
		int n = xStatistic.get(0).length; //1, x1, x2,..., x(n-1)
		List<Double> alpha = calcCoeffsByStatistics(xStatistic, zDataToLearnAlpha);
		if (alpha == null || alpha.size() == 0) { //If cannot calculate alpha by matrix calculation.
			if (currentParameter != null)
				alpha = DSUtil.toDoubleList(currentParameter.alpha); //clone alpha
			else { //Used for initialization so that regression model is always determined.
				alpha = DSUtil.initDoubleList(n, 0.0);
				double alpha0 = 0;
				for (int i = 0; i < N; i++)
					alpha0 += zStatistic.get(i)[1];
				alpha.set(0, alpha0 / (double)N); //constant function z = c
			}
		}
		else if (Util.isUsed(zFactor)){
			if (zFactor == 0)
				zFactor = Double.MIN_VALUE;
			for (int j = 0; j < alpha.size(); j++) {
				alpha.set(j, alpha.get(j) / zFactor); //Adjusting alpha
			}
		}
		
		List<double[]> betas = Util.newList(n);
		for (int j = 0; j < n; j++) {
			if (j == 0) {
				double[] beta0 = new double[2];
				beta0[0] = 1;
				beta0[1] = 0;
				betas.add(beta0);
				continue;
			}
			
			List<double[]> Z = Util.newList(N);
			List<Double> x = Util.newList(N);
			for (int i = 0; i < N; i++) {
				Z.add(zStatistic.get(i));
				x.add(xStatistic.get(i)[j]);
			}
			List<Double> beta = calcCoeffs(Z, x);
			if (beta == null || beta.size() == 0) {
				if (currentParameter != null)
					beta = DSUtil.toDoubleList(currentParameter.getBetas().get(j));
				else { //Used for initialization so that regression model is always determined.
					beta = DSUtil.initDoubleList(2, 0);
					double beta0 = 0;
					for (int i = 0; i < N; i++)
						beta0 += xStatistic.get(i)[j];
					beta.set(0, beta0 / (double)N); //constant function x = c
				}
			}
			betas.add(DSUtil.toDoubleArray(beta));
		}
		
		return new ExchangedParameter(alpha, betas);
	}

	
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		ExchangedParameter parameter = (ExchangedParameter)super.initializeParameter();
		if (parameter == null)
			return null;
		
		this.mean0 = getConfig().getAsReal(MEAN0_FIELD);
		this.variance0 = getConfig().getAsReal(VARIANCE0_FIELD);
		
		if (Util.isUsed(this.mean0) && Util.isUsed(this.variance0)) {
			return parameter;
		}
		
		LargeStatistics stat = getLargeStatistics();
		List<double[]> zData = stat.getZData();
		
		double sum = 0;
		int N = 0;
		for (int i = 0; i < zData.size(); i++) {
			double zValue = zData.get(i)[1];
			if (Util.isUsed(zValue)) {
				sum += zValue;
				N++;
			}
		}
		this.mean0 = sum / N;
		
		double ss = 0;
		for (int i = 0; i < zData.size(); i++) {
			double zValue = zData.get(i)[1];
			if (Util.isUsed(zValue)) {
				double d = zValue - this.mean0;
				ss += d*d;
			}
		}
		this.variance0 = ss / (N*N);
		
		return parameter;
	}


	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(MEAN0_FIELD, MEAN0_DEFAULT);
		config.put(VARIANCE0_FIELD, VARIANCE0_DEFAULT);
		return config;
	}

	
}
