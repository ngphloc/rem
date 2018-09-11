package net.hudup.regression.em;

import static net.hudup.regression.AbstractRegression.defaultExtractVariable;
import static net.hudup.regression.em.RegressionEMImpl.R_CALC_VARIANCE_FIELD;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.DSUtil;

/**
 * This class implements the mixture regression model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DefaultMixtureRegressionEM extends AbstractMixtureRegressionEM implements DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of cluster number field.
	 */
	public final static String COMP_NUMBER_FIELD = "comp_number";

	
	/**
	 * Default number of cluster.
	 */
	public final static int COMP_NUMBER_DEFAULT = 0;

	
	/**
	 * Internal data.
	 */
	protected LargeStatistics data = null;
	
	
	/**
	 * Indices for X data.
	 */
	protected List<Object[]> xIndices = Util.newList(); //Object list for parsing mathematical expressions in the most general case.
	
	
	/**
	 * Indices for Z data.
	 */
	protected List<Object[]> zIndices = Util.newList(); //Object list for parsing mathematical expressions in the most general case.
	
	
	/**
	 * Attribute list for all variables: all X, Y, and z.
	 * This variable is also used as the indicator of successful learning (not null).
	 */
	protected AttributeList attList = null;

	
	@Override
	public synchronized void unsetup() {
		// TODO Auto-generated method stub
		super.unsetup();
		
		if (this.data != null)
			this.data.clear();
		this.data = null;
	}


	@Override
	protected boolean prepareInternalData(Fetcher<Profile> inputSample) throws Exception {
		clearInternalContent();

		RegressionEMImpl tempEM = new RegressionEMImpl();
		tempEM.getConfig().put(R_INDICES_FIELD, this.getConfig().get(R_INDICES_FIELD));
		if (!tempEM.prepareInternalData(inputSample))
			return false;
		
		this.xIndices = tempEM.xIndices;
		this.zIndices = tempEM.zIndices;
		this.attList = tempEM.attList;
		this.data = tempEM.data; tempEM.data = null;
		
		int K = getConfig().getAsInt(COMP_NUMBER_FIELD);
		K = K <= 0 ? 3 : K; //Improve here, getting exact the number of PRMs 
		this.rems = Util.newList(K);
		for (int k = 0; k < K; k++) {
			RegressionEMImpl rem = createRegressionEM();
			rem.prepareInternalData(this.xIndices, this.zIndices, this.attList, this.data);
			this.rems.add(rem);
		}
		
		return true;
	}

	
	@Override
	protected void clearInternalContent() {
		// TODO Auto-generated method stub
		super.clearInternalContent();
		this.xIndices.clear();
		this.zIndices.clear();
		this.attList = null;
	}


	@Override
	protected RegressionEMImpl createRegressionEM() {
		// TODO Auto-generated method stub
		RegressionEMExt rem = new RegressionEMExt();
		rem.getConfig().put(R_CALC_VARIANCE_FIELD, true);
		return rem;
	}


	@Override
	protected Object expectation(Object currentParameter, Object... info) throws Exception {
		// TODO Auto-generated method stub
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> parameters = (List<ExchangedParameter>)currentParameter;
		@SuppressWarnings("unchecked")
		List<LargeStatistics> stats = (List<LargeStatistics>)super.expectation(currentParameter, info);
		
		//Adjusting large statistics.
		int N = stats.get(0).getZData().size(); //Suppose all models have the same data.
		int n = stats.get(0).getXData().get(0).length;  //Suppose all models have the same data.
		List<double[]> xData = Util.newList(N);
		List<double[]> zData = Util.newList(N);
		for (int i = 0; i < N; i++) {
			double[] xVector = new double[n];
			Arrays.fill(xVector, 0.0);
			xVector[0] = 1;
			xData.add(xVector);
			
			double[] zVector = new double[2];
			zVector[0] = 1;
			zVector[1] = 0;
			zData.add(zVector);
		}
		for (int k = 0; k < this.rems.size(); k++) {
			double coeff = parameters.get(k).getCoeff();
			LargeStatistics stat = stats.get(k);
			
			for (int i = 0; i < N; i++) {
				double[] xVector = xData.get(i);
				for (int j = 1; j < n; j++) {
					double xValue = stat.getXData().get(i)[j];
					xVector[j] += coeff * xValue; // This assignment is not totally exact. In next version, inverse regression model is also associated mixture model. 
				}
				
				double[] zVector = zData.get(i);
				double zValue = stat.getZData().get(i)[1];
				zVector[1] += coeff * zValue;
			}
		}

		//All regression models have the same large statistics.
		stats.clear();
		LargeStatistics stat = new LargeStatistics(xData, zData);
		for (RegressionEMImpl rem : this.rems) {
			rem.setStatistics(stat);
			stats.add(stat);
		}
		
		return stats;
	}


	@Override
	protected Object maximization(Object currentStatistic, Object... info) throws Exception {
		// TODO Auto-generated method stub
		if (currentStatistic == null)
			return null;
		@SuppressWarnings("unchecked")
		List<LargeStatistics> stats = (List<LargeStatistics>)currentStatistic;
		List<ExchangedParameter> parameters = Util.newList(this.rems.size());
		List<List<Double>> condProbs = Util.newList(this.rems.size()); //K lists of conditional probabilities.
		
		int K = this.rems.size();
		for (int k = 0; k < K; k++) {
			LargeStatistics stat = stats.get(k); //Each REM has particular large statistics. 
			int N = stat.getZStatistic().size();
			List<Double> kCondProbs = Util.newList(N); //The kth list of conditional probabilities.
			condProbs.add(kCondProbs);
			
			for (int i = 0; i < N; i++) {
				List<double[]> xData = Util.newList(K);
				List<double[]> zData = Util.newList(K);
				
				for (int j = 0; j < K; j++) {
					xData.add(stat.getXData().get(i));
					zData.add(stat.getZData().get(i));
				}
				
				@SuppressWarnings("unchecked")
				List<Double> probs = condZProbs((List<ExchangedParameter>)this.getCurrentParameter(),
						xData, zData);
				kCondProbs.add(probs.get(k));
			}
		}
		
		for (int k = 0; k < K; k++) {
			RegressionEMImpl rem = this.rems.get(k);
			LargeStatistics stat = stats.get(k);

			ExchangedParameter parameter = (ExchangedParameter)rem.maximization(stat, condProbs.get(k));
			rem.setEstimatedParameter(parameter);
			parameters.add(parameter);
		}
		
		return parameters;
	}


	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		List<ExchangedParameter> parameters = Util.newList(this.rems.size());
		LargeStatistics completeData = RegressionEMImpl.getCompleteData(this.data);
		int recordNumber = (completeData != null && completeData.getZData().size() >= this.rems.size())? 
				completeData.getZData().size() / this.rems.size() :
				0;
		
		for (int k = 0; k < this.rems.size(); k++) {
			RegressionEMImpl rem = this.rems.get(k);
			ExchangedParameter parameter = null;
			LargeStatistics compSample = null;
			if (recordNumber > 0) {
				compSample = randomSampling(completeData, recordNumber, false);
				try {
					parameter = (ExchangedParameter) rem.maximization(compSample);
				}
				catch (Throwable e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					parameter = null;
				}
			}
			if (parameter == null) {
				parameter = RegressionEMImpl.initializeAlphaBetas(this.data.getXData().get(0).length, true);
				parameter.setZVariance(1.0);
			}
			parameter.setCoeff(1.0 / (double)this.rems.size());
			
			rem.setEstimatedParameter(parameter);
			rem.setCurrentParameter(parameter);
			rem.setPreviousParameter(null);
			rem.setStatistics(null);
			rem.setCurrentIteration(this.getCurrentIteration());

			parameters.add(parameter);
		}
		
		return parameters;
	}


	/**
	 * This class is an extension of regression expectation maximization algorithm.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	protected class RegressionEMExt extends RegressionEMImpl {
		
		/**
		 * Serial version UID for serializable class.
		 */
		private static final long serialVersionUID = 1L;
		
		@Override
		protected Object maximization(Object currentStatistic, Object... info) throws Exception {
			// TODO Auto-generated method stub
			LargeStatistics stat = (LargeStatistics)currentStatistic;
			if (stat == null || stat.isEmpty())
				return null;
			List<double[]> xStatistic = stat.getXData();
			List<double[]> zStatistic = stat.getZData();
			int N = zStatistic.size();
			int n = xStatistic.get(0).length; //1, x1, x2,..., x(n-1)
			ExchangedParameter currentParameter = (ExchangedParameter)getCurrentParameter();
			
			List<double[]> uStatistic = xStatistic;
			List<double[]> vStatistic = zStatistic;
			List<Double> kCondProbs = null;
			if (info != null && info.length > 0 && (info[0] instanceof List<?>)) {
				@SuppressWarnings("unchecked")
				List<Double> kCondProbTemp = (List<Double>)info[0];
				kCondProbs = kCondProbTemp;
				
				uStatistic = Util.newList(xStatistic.size());
				vStatistic = Util.newList(zStatistic.size());
				for (int i = 0; i < N; i++) {
					double[] uVector = new double[n];
					uStatistic.add(uVector);
					double[] vVector = new double[2];
					vStatistic.add(vVector);
					
					for (int j = 0; j < n; j++) {
						uVector[j] = xStatistic.get(i)[j] * kCondProbs.get(i); 
					}
					vVector[0] = 1;
					vVector[1] = zStatistic.get(i)[1] * kCondProbs.get(i); 
				}
			}
			
			List<Double> alpha = calcCoeffsByStatistics(uStatistic, vStatistic);
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
			if (kCondProbs == null) {
				newParameter.setZVariance(newParameter.estimateZVariance(stat));
			}
			else {
				double sumCondProb = 0;
				for (int i = 0; i < N; i++) {
					sumCondProb += kCondProbs.get(i);
				}
				
				double sumZVariance = 0;
				for (int i = 0; i < N; i++) {
					double d = zStatistic.get(i)[1] - ExchangedParameter.mean(alpha, xStatistic.get(i));
					sumZVariance += d*d*kCondProbs.get(i);
				}
				
				newParameter.setCoeff(sumCondProb/N);
				if (sumCondProb != 0)
					newParameter.setZVariance(sumZVariance/sumCondProb);
				else
					newParameter.setZVariance(1.0); //Fixing zero probabilities.
			}
			
			return newParameter;
		}

		@Override
		protected double extractRegressor(Object input, int index) {
			// TODO Auto-generated method stub
			return getDefaultMREM().extractRegressor(input, index);
		}

		@Override
		protected Object transformRegressor(Object x, boolean inverse) {
			// TODO Auto-generated method stub
			return getDefaultMREM().transformRegressor(x, inverse);
		}

		@Override
		protected Object transformResponse(Object z, boolean inverse) {
			// TODO Auto-generated method stub
			return getDefaultMREM().transformResponse(z, inverse);
		}
		
	}
	
	
	/**
	 * Getting this mixture regression expectation maximization model.
	 * @return this mixture regression expectation maximization model.
	 */
	private DefaultMixtureRegressionEM getDefaultMREM() {
		return this;
	}
	
	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "default_mixrem";
	}

	
	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		DefaultMixtureRegressionEM mixREM = new DefaultMixtureRegressionEM();
		mixREM.getConfig().putAll((DataConfig)this.getConfig().clone());
		return mixREM;
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(R_INDICES_FIELD, R_INDICES_DEFAULT);
		config.put(COMP_NUMBER_FIELD, COMP_NUMBER_DEFAULT);
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	
	/**
	 * Extracting value of regressor (X) from specified profile.
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param input specified input. It is often profile.
	 * @param index specified index. Index 0 is not included in the profile because this specified index is in internal indices.
	 * So index 0 always indicates to value 1. 
	 * @return value of regressor (X) extracted from specified profile.
	 */
	protected double extractRegressor(Object input, int index) {
		// TODO Auto-generated method stub
		if (input == null)
			return Constants.UNUSED;
		else if (input instanceof Profile)
			return defaultExtractVariable(input, null, xIndices, index);
		else
			return defaultExtractVariable(input, attList, xIndices, index);
	}

	
	/**
	 * Transforming independent variable X.
	 * In the most general case that each index is an mathematical expression, this method is not focused.
	 * @param x specified variable X.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of X.
	 */
	protected Object transformRegressor(Object x, boolean inverse) {
		// TODO Auto-generated method stub
		return x;
	}

	
	/**
	 * Transforming independent variable Z.
	 * In the most general case that each index is an mathematical expression, this method is not focused but is useful in some cases.
	 * @param z specified variable Z.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of Z.
	 */
	protected Object transformResponse(Object z, boolean inverse) {
		// TODO Auto-generated method stub
		return z;
	}
	
	
	/**
	 * Randomized sampling the specified data.
	 * @param data the specified data.
	 * @param recordNumber the number of randomized records.
	 * @param giveBack if true, the random record is given back to original sample.
	 * @return Randomized sample the specified data.
	 */
	private static LargeStatistics randomSampling(LargeStatistics data, int recordNumber, boolean giveBack) {
		if (data.getZData().size() == 0 || recordNumber <=0 )
			return null;
		
		List<double[]> xData = Util.newList();
		List<double[]> zData = Util.newList();
		Random rnd = new Random();
		for (int i = 0; i < recordNumber; i++) {
			int N = data.getZData().size();
			if (N == 0)
				break;
			int j = rnd.nextInt(N);
			xData.add(data.getXData().get(j));
			zData.add(data.getZData().get(j));
			
			if (!giveBack) {
				data.getXData().remove(j);
				data.getZData().remove(j);
			}
		}
		
		if (zData.size() == 0)
			return null;
		else
			return new LargeStatistics(xData, zData);
	}
	
	
}
