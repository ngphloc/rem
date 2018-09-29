package net.hudup.regression.em;

import java.util.Arrays;
import java.util.List;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.data.DataConfig;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.parser.TextParserUtil;

/**
 * This class implements expectation maximization algorithm for regression model in case of missing data with support of the prior probability.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class REMPrior extends REMImpl {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of alpha coefficients of prior distribution.
	 */
	public static final String ALPHA0_FIELD = "alpha0";

			
	/**
	 * Default alpha coefficients of prior distribution.
	 */
	public static final String ALPHA0_DEFAULT = "";

	
	/**
	 * Name of variance of prior distribution.
	 */
	public static final String ZVARIANCE0_FIELD = "zvariance0";

			
	/**
	 * Default variance of prior distribution.
	 */
	public static final double ZVARIANCE0_DEFAULT = Constants.UNUSED;

	
	/**
	 * Alpha coefficients of prior distribution.
	 */
	protected List<Double> alpha0 = null;
	
	
	/**
	 * Variance of prior distribution.
	 */
	protected double zVariance0 = 0;

	
	/**
	 * Default constructor.
	 */
	public REMPrior() {
		// TODO Auto-generated constructor stub
		super();
	}
	
	
	@Override
	protected void clearInternalData() {
		// TODO Auto-generated method stub
		super.clearInternalData();
		this.alpha0 = null;
		this.zVariance0 = 0;
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
		REMPrior priorREM = new REMPrior();
		priorREM.getConfig().putAll((DataConfig)this.getConfig().clone());
		return priorREM;
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
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		ExchangedParameter parameter = (ExchangedParameter)super.initializeParameter();
		if (parameter == null)
			return null;
		
		String alpha0Text = getConfig().getAsString(ALPHA0_FIELD);
		if (alpha0Text == null || alpha0Text.isEmpty())
			this.alpha0 = parameter.getAlpha();
		else
			this.alpha0 = TextParserUtil.parseListByClass(alpha0Text, Double.class, ",");
		if (this.alpha0 == null || this.alpha0.size() == 0)
			return null;
		
		this.zVariance0 = getConfig().getAsReal(ZVARIANCE0_FIELD);
		if (!Util.isUsed(this.zVariance0)) {
			LargeStatistics stat;
			try {
				stat = (LargeStatistics)this.expectation(parameter);
				this.zVariance0 = parameter.estimateZVariance(stat) / stat.getZData().size();
			} 
			catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				return null;
			}
		}
		
		return parameter;
	}


	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
		return super.getDescription() + ", " + moreParametersToText();
	}


	@Override
	public String parameterToShownText(Object parameter, Object... info) {
		// TODO Auto-generated method stub
		return super.parameterToShownText(parameter, info) + ", " + moreParametersToText();
	}


	/**
	 * Converting prior alpha and prior variance to text.
	 * @return text of prior alpha and prior variance
	 */
	private String moreParametersToText() {
		StringBuffer buffer = new StringBuffer();
		
		if (this.alpha0 == null)
			buffer.append("alpha0=()");
		else {
			buffer.append("alpha0=(");
			for (int j = 0; j < this.alpha0.size(); j++) {
				if (j > 0)
					buffer.append(", ");
				buffer.append(MathUtil.format(this.alpha0.get(j)));
			}
			buffer.append(")");
		}
		
		buffer.append(", z-variance0=" + MathUtil.format(this.zVariance0));
		
		return buffer.toString();
	}
	
	
	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(ALPHA0_FIELD, ALPHA0_DEFAULT);
		config.put(ZVARIANCE0_FIELD, ZVARIANCE0_DEFAULT);
		return config;
	}

	
}
