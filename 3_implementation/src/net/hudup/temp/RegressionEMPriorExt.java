package net.hudup.temp;

import java.util.Arrays;
import java.util.List;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.data.DataConfig;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.NextUpdate;
import net.hudup.regression.em.ExchangedParameter;
import net.hudup.regression.em.LargeStatistics;
import net.hudup.regression.em.RegressionEMPrior;

/**
 * This class represents an extension of regression expectation maximization algorithm with support of prior probability.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@NextUpdate
public class RegressionEMPriorExt extends RegressionEMPrior {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public RegressionEMPriorExt() {
		// TODO Auto-generated constructor stub
		super();
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
			double zVariance = currentParameter.estimateZVariance(stat);
			zFactor = this.zVariance0 - zVariance;
			
			for (int i = 0; i < N; i++) {
				double[] xVector =  xStatistic.get(i);
				double[] zVector = Arrays.copyOf(zStatistic.get(i), zStatistic.get(i).length); 
				double zMean0 = ExchangedParameter.mean(this.alpha0, xVector);
				zVector[1] = this.zVariance0*zVector[1] - zVariance*zMean0;
				
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
				alpha = DSUtil.toDoubleList(currentParameter.getAlpha()); //clone alpha
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
		
		ExchangedParameter newParameter = new ExchangedParameter(alpha, betas);
		if (getConfig().getAsBoolean(R_CALC_VARIANCE_FIELD))
			newParameter.setZVariance(newParameter.estimateZVariance(stat));
		else {
			if (currentParameter == null)
				newParameter.setZVariance(Constants.UNUSED);
			else if (Util.isUsed(currentParameter.getZVariance()))
				newParameter.setZVariance(newParameter.estimateZVariance(stat));
			else
				newParameter.setZVariance(Constants.UNUSED);
		}
		
		return newParameter;
	}

	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "prior_rem_ext";
	}

	
	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		RegressionEMPriorExt priorREMExt = new RegressionEMPriorExt();
		priorREMExt.getConfig().putAll((DataConfig)this.getConfig().clone());
		return priorREMExt;
	}
	
	
}
