package net.hudup.regression.em;

import java.util.List;

import net.hudup.core.Constants;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Profile;
import net.hudup.em.ExponentialEM;
import net.hudup.regression.em.DefaultRegressionEM.ExchangedParameter;


/**
 * This class implements expectation maximization (EM) algorithm for two dual regression models when the response variable of the second model can be missing. 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DualRegressionEM extends ExponentialEM implements RegressionEM, DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Regression indices field.
	 */
	public final static String REM_INDICES_FIELD2 = "rem_indices2";

	
	/**
	 * Name of first execution model field.
	 */
	protected final static String EXECUTION_FIRST_MODE_FIELD = "drem_execution_first_model";
	
	
	/**
	 * Default first execution model.
	 */
	protected final static boolean EXECUTION_FIRST_MODE_DEFAULT = true;

	
	/**
	 * The first model.
	 */
	protected DefaultRegressionEM rem1 = null;
	
	
	/**
	 * The second model.
	 */
	protected DefaultRegressionEM rem2 = null;

	
	/**
	 * Default constructor.
	 */
	public DualRegressionEM() {
		super();
		// TODO Auto-generated constructor stub
	}


	@Override
	public synchronized Object learn() throws Exception {
		// TODO Auto-generated method stub
		if (!prepareInternalData())
			return null;
		
		ExchangedParameter[] parameters = (ExchangedParameter[])super.learn();
		if (parameters == null || parameters[0] == null )
			return null;
		if (parameters[0] == null && parameters[1] == null)
			return null;

		if (this.rem1 != null && parameters[0] != null)
			this.rem1.setParameter(parameters[0], 0);
		if (this.rem2 != null && parameters[1] != null)
			this.rem2.setParameter(parameters[1], 0);
		
		return parameters;
	}


	@Override
	public synchronized void unsetup() {
		// TODO Auto-generated method stub
		super.unsetup();
		if (this.rem1 != null)
			this.rem1.unsetup();
		if (this.rem2 != null)
			this.rem2.unsetup();
	}

	
	/**
	 * Preparing data.
	 * @return true if data preparation is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean prepareInternalData() throws Exception {
		clearInternalData();
		DataConfig thisConfig = this.getConfig();
		
		DefaultRegressionEM rem1 = new DefaultRegressionEM() {

			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public synchronized Object learn() throws Exception {
				// TODO Auto-generated method stub
				return prepareInternalData();
			}

			@Override
			protected double extractRegressor(Profile profile, int index) {
				// TODO Auto-generated method stub
				return getThis().extractRegressor(profile, xIndices, index);
			}

			@Override
			protected String extractRegressorText(int index) {
				// TODO Auto-generated method stub
				return getThis().extractRegressorText(attList, xIndices, index);
			}

			@Override
			public double extractResponse(Profile profile) {
				// TODO Auto-generated method stub
				return getThis().extractResponse(profile, zIndices);
			}

			@Override
			protected String extractResponseText() {
				// TODO Auto-generated method stub
				return getThis().extractResponseText(attList, zIndices);
			}

			@Override
			protected Object transformRegressor(Object x) {
				// TODO Auto-generated method stub
				return getThis().transformRegressor(x, true);
			}

			@Override
			protected Object inverseTransformRegressor(Object inverseX) {
				// TODO Auto-generated method stub
				return getThis().inverseTransformRegressor(inverseX, true);
			}

			@Override
			protected Object transformResponse(Object z) {
				// TODO Auto-generated method stub
				return getThis().transformResponse(z, true);
			}

			@Override
			protected Object inverseTransformResponse(Object inverseZ) {
				// TODO Auto-generated method stub
				return getThis().inverseTransformResponse(inverseZ, true);
			}
			
		};
		DataConfig config1 = rem1.getConfig();
		config1.put(DefaultRegressionEM.REM_INDICES_FIELD, thisConfig.get(DefaultRegressionEM.REM_INDICES_FIELD));
		config1.put(DefaultRegressionEM.REM_INVERSE_MODE_FIELD, thisConfig.get(DefaultRegressionEM.REM_INVERSE_MODE_FIELD));
		config1.put(DefaultRegressionEM.REM_BALANCE_MODE_FIELD, thisConfig.get(DefaultRegressionEM.REM_BALANCE_MODE_FIELD));
		rem1.setup(this.dataset);
		if(rem1.attList != null)
			this.rem1 = rem1;
		else
			rem1.clearInternalData();
		
		DefaultRegressionEM rem2 = new DefaultRegressionEM() {

			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public synchronized Object learn() throws Exception {
				// TODO Auto-generated method stub
				return prepareInternalData();
			}

			@Override
			protected double extractRegressor(Profile profile, int index) {
				// TODO Auto-generated method stub
				return getThis().extractRegressor(profile, xIndices, index);
			}

			@Override
			protected String extractRegressorText(int index) {
				// TODO Auto-generated method stub
				return getThis().extractRegressorText(attList, xIndices, index);
			}

			@Override
			public double extractResponse(Profile profile) {
				// TODO Auto-generated method stub
				return getThis().extractResponse(profile, zIndices);
			}

			@Override
			protected String extractResponseText() {
				// TODO Auto-generated method stub
				return getThis().extractResponseText(attList, zIndices);
			}

			@Override
			protected Object transformRegressor(Object x) {
				// TODO Auto-generated method stub
				return getThis().transformRegressor(x, false);
			}

			@Override
			protected Object inverseTransformRegressor(Object inverseX) {
				// TODO Auto-generated method stub
				return getThis().inverseTransformRegressor(inverseX, false);
			}

			@Override
			protected Object transformResponse(Object z) {
				// TODO Auto-generated method stub
				return getThis().transformResponse(z, false);
			}

			@Override
			protected Object inverseTransformResponse(Object inverseZ) {
				// TODO Auto-generated method stub
				return getThis().inverseTransformResponse(inverseZ, false);
			}
			
		};
		DataConfig config2 = rem2.getConfig();
		config2.put(REM_INDICES_FIELD2, thisConfig.get(REM_INDICES_FIELD2));
		config2.put(DefaultRegressionEM.REM_INVERSE_MODE_FIELD, thisConfig.get(DefaultRegressionEM.REM_INVERSE_MODE_FIELD));
		config2.put(DefaultRegressionEM.REM_BALANCE_MODE_FIELD, thisConfig.get(DefaultRegressionEM.REM_BALANCE_MODE_FIELD));
		rem2.setup(this.dataset);
		if(rem2.attList != null)
			this.rem2 = rem2;
		else
			rem2.clearInternalData();

		if (this.rem1 == null && this.rem2 == null) {
			clearInternalData();
			return false;
		}
		else
			return true;
	}
	
	
	/**
	 * Clear all internal data.
	 */
	protected void clearInternalData() {
		this.currentIteration = 0;
		this.currentParameter = this.estimatedParameter = null;
		if (this.rem1 != null) {
			this.rem1.clearInternalData();
			this.rem1 = null;
		}
		if (this.rem2 != null) {
			this.rem2.clearInternalData();
			this.rem2 = null;
		}
	}

	
	/**
	 * Getting this dual regression model.
	 * @return this dual regression model.
	 */
	private DualRegressionEM getThis() {
		return this;
	}
	
	
	@Override
	protected Object expectation(Object currentParameter) throws Exception {
		// TODO Auto-generated method stub
		ExchangedParameter parameter1 = ((ExchangedParameter[])currentParameter)[0];
		ExchangedParameter parameter2 = ((ExchangedParameter[])currentParameter)[1];
		
		ExchangedParameter stat1 = null;
		if (parameter1 != null && this.rem1 != null)
			stat1 = (ExchangedParameter)this.rem1.expectation(parameter1);
		
		ExchangedParameter stat2 = null;
		if (parameter2 != null && this.rem2 != null)
			stat2 = (ExchangedParameter)this.rem2.expectation(parameter2);

		if(stat1 != null && stat2 != null) {
			stat1 = stat1.mean(stat2);
			stat2 = stat1;
		}
		
		if (stat1 == null && stat2 == null)
			return null;
		else
			return new ExchangedParameter[] {stat1, stat2};
	}

	
	@Override
	protected Object maximization(Object currentStatistic) throws Exception {
		// TODO Auto-generated method stub
		ExchangedParameter stat1 = ((ExchangedParameter[])currentStatistic)[0];
		ExchangedParameter stat2 = ((ExchangedParameter[])currentStatistic)[1];

		ExchangedParameter parameter1 = null;
		if (stat1 != null && this.rem1 != null)
			parameter1 = (ExchangedParameter)this.rem1.maximization(stat1);
		
		ExchangedParameter parameter2 = null;
		if (stat2 != null && this.rem2 != null)
			parameter2 = (ExchangedParameter)this.rem2.maximization(stat2);

		if (parameter1 == null && parameter2 == null)
			return null;
		else
			return new ExchangedParameter[] {parameter1, parameter2};
	}

	
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		ExchangedParameter parameter1 = null;
		if (this.rem1 != null)
			parameter1 = (ExchangedParameter)this.rem1.initializeParameter();
		
		ExchangedParameter parameter2 = null;
		if (this.rem2 != null)
			parameter2 = (ExchangedParameter)this.rem2.initializeParameter();

		return new ExchangedParameter[] {parameter1, parameter2};
	}

	
	@Override
	protected boolean terminatedCondition(Object currentParameter, Object estimatedParameter, Object... info) {
		// TODO Auto-generated method stub
		boolean terminated = true;

		ExchangedParameter param1 = ((ExchangedParameter[])currentParameter)[0];
		ExchangedParameter estimatedParam1 = ((ExchangedParameter[])estimatedParameter)[0];
		if (param1 != null && estimatedParam1 != null && this.rem1 != null)
			terminated = terminated && this.rem1.terminatedCondition(param1, estimatedParam1, info);

		if (!terminated)
			return false;
		
		ExchangedParameter param2 = ((ExchangedParameter[])currentParameter)[1];
		ExchangedParameter estimatedParam2 = ((ExchangedParameter[])estimatedParameter)[1];
		if (param2 != null && estimatedParam2 != null && this.rem2 != null)
			terminated = terminated && this.rem2.terminatedCondition(param2, estimatedParam2, info);

		return terminated;
	}

	
	@Override
	public Object execute(Object input) {
		// TODO Auto-generated method stub
		boolean executionMode = getConfig().getAsBoolean(EXECUTION_FIRST_MODE_FIELD);
		DefaultRegressionEM rem = null;
		if (executionMode)
			rem = this.rem1;
		else
			rem = this.rem2;
		
		if (rem == null)
			return null;
		else
			return rem.execute(input);
	}

	
	@Override
	public String parameterToShownText(Object parameter, Object... info) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof ExchangedParameter[]))
			return "";
		StringBuffer buffer = new StringBuffer();
		
		if (this.rem1 != null && ((ExchangedParameter[])parameter)[0] != null) {
			String text1 = this.rem1.parameterToShownText(((ExchangedParameter[])parameter)[0], info);
			buffer.append(text1);
		}
		
		if (this.rem2 != null && ((ExchangedParameter[])parameter)[1] != null) {
			String text2 = this.rem2.parameterToShownText(((ExchangedParameter[])parameter)[1], info);
			if (buffer.length() > 0)
				buffer.append(", ");
			buffer.append(text2);
		}

		return buffer.toString();
	}

	
	@Override
	public String getDescription() {
		// TODO Auto-generated method stub
		StringBuffer buffer = new StringBuffer();

		if (this.rem1 != null) {
			String text1 = this.rem1.getDescription();
			buffer.append(text1);
		}

		if (this.rem2 != null) {
			String text2 = this.rem2.getDescription();
			if (buffer.length() > 0)
				buffer.append(", ");
			buffer.append(text2);
		}

		return null;
	}

	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "drem";
	}

	
	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		DualRegressionEM drem = new DualRegressionEM();
		drem.getConfig().putAll((DataConfig)this.getConfig().clone());
		return drem;
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
		config.put(DefaultRegressionEM.REM_INDICES_FIELD, "{-1}, {-1}, {-1}"); //Not used
		config.put(DefaultRegressionEM.REM_INVERSE_MODE_FIELD, DefaultRegressionEM.REM_INVERSE_MODE_DEFAULT);
		config.put(DefaultRegressionEM.REM_BALANCE_MODE_FIELD, DefaultRegressionEM.REM_BALANCE_MODE_DEFAULT);
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		
		config.put(REM_INDICES_FIELD2, "{-1}, {-1}, {-1}"); //Not used
		config.put(EXECUTION_FIRST_MODE_FIELD, EXECUTION_FIRST_MODE_DEFAULT); // execution mode
		return config;
	}

	
	/**
	 * Extracting value of regressor (X) from specified profile.
	 * @param profile specified profile.
	 * @param xIndices specified indices of regressors.
	 * @param index specified indices.
	 * @return value of regressor (X) extracted from specified profile.
	 */
	protected double extractRegressor(Profile profile, List<int[]> xIndices, int index) {
		// TODO Auto-generated method stub
		return profile.getValueAsReal(xIndices.get(index)[0]);
	}

	
	/**
	 * Extracting text of response variable (Z).
	 * @param attList specified attribute list.
	 * @param xIndices specified indices of regressors.
	 * @param index specified index.
	 * @return text of response variable (Z) extracted.
	 */
	protected String extractRegressorText(AttributeList attList, List<int[]> xIndices, int index) {
		// TODO Auto-generated method stub
		return attList.get(xIndices.get(index)[0]).getName();
	}

	
	@Override
	public double extractResponse(Profile profile) {
		// TODO Auto-generated method stub
		boolean executionMode = getConfig().getAsBoolean(EXECUTION_FIRST_MODE_FIELD);
		DefaultRegressionEM rem = null;
		if (executionMode)
			rem = this.rem1;
		else
			rem = this.rem2;
		
		if (rem == null)
			return Constants.UNUSED;
		else
			return extractResponse(profile, rem.zIndices);
	}

	
	/**
	 * Extracting value of response variable (Z) from specified profile and indices.
	 * @param profile specified profile.
	 * @param zIndices specified indices of response variables.
	 * @return value of response variable (Z) extracted from specified profile and indices.
	 */
	protected double extractResponse(Profile profile, List<int[]> zIndices) {
		// TODO Auto-generated method stub
		return profile.getValueAsReal(zIndices.get(1)[0]);
	}

	
	/**
	 * Extracting text of response variable (Z).
	 * @param attList specified attribute list.
	 * @param zIndices specified indices of response variables.
	 * @return text of response variable (Z) extracted.
	 */
	protected String extractResponseText(AttributeList attList, List<int[]> zIndices) {
		// TODO Auto-generated method stub
		return attList.get(zIndices.get(1)[0]).getName();
	}

	
	/**
	 * Transforming independent variable X.
	 * @param x specified variable X.
	 * @param firstModel if true, the first model is used.
	 * @return transformed value of X.
	 */
	protected Object transformRegressor(Object x, boolean firstModel) {
		// TODO Auto-generated method stub
		if (x == null)
			return null;
		else if (x instanceof Number)
			return ((Number)x).doubleValue();
		else if (x instanceof String)
			return (String)x;
		else
			return x;
	}

	
	/**
	 * Inverse transforming of the inverse value of independent variable X.
	 * This method is the inverse of {@link #transformRegressor(double)}.
	 * @param inverseX inverse value of independent variable X.
	 * @param firstModel if true, the first model is used.
	 * @return value of X.
	 */
	protected Object inverseTransformRegressor(Object inverseX, boolean firstModel) {
		// TODO Auto-generated method stub
		return transformRegressor(inverseX, firstModel);
	}

	
	/**
	 * Transforming independent variable Z.
	 * @param z specified variable Z.
	 * @param firstModel if true, the first model is used.
	 * @return transformed value of Z.
	 */
	protected Object transformResponse(Object z, boolean firstModel) {
		// TODO Auto-generated method stub
		if (z == null)
			return null;
		else if (z instanceof Number)
			return ((Number)z).doubleValue();
		else if (z instanceof String)
			return (String)z;
		else
			return z;
	}

	
	/**
	 * Inverse transforming of the inverse value of independent variable Z.
	 * This method is the inverse of {@link #transformResponse(double)}.
	 * @param inverseZ inverse value of independent variable Z.
	 * @param firstModel if true, the first model is used.
	 * @return value of Z.
	 */
	protected Object inverseTransformResponse(Object inverseZ, boolean firstModel) {
		// TODO Auto-generated method stub
		return transformResponse(inverseZ, firstModel);
	}


}
