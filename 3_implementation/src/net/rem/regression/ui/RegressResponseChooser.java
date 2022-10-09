/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression.ui;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;

import net.hudup.core.Util;
import net.hudup.core.data.Attribute;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.Wrapper;
import net.hudup.core.logistic.ui.JCheckList;
import net.hudup.core.logistic.ui.JRadioList;
import net.hudup.core.logistic.ui.UIUtil;


/**
 * This class is the dialog which allows users to select regressors (independent variable) and response (dependent variable).
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RegressResponseChooser extends JDialog {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of returned regressors.
	 */
	protected List<Attribute> selectedRegressors = Util.newList();
	
	
	/**
	 * Returned responsor.
	 */
	protected Attribute selectedResponsor = null;

	
	/**
	 * Constructor with parent component and attribute list.
	 * @param comp parent component.
	 * @param attList attribute list.
	 */
	public RegressResponseChooser(Component comp, AttributeList attList) {
		// TODO Auto-generated constructor stub
		super(UIUtil.getDialogForComponent(comp), "Choosing algorithms", true);
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		setSize(400, 300);
		setLocationRelativeTo(UIUtil.getDialogForComponent(comp));
		
		setLayout(new BorderLayout());
		JPanel body = new JPanel(new GridLayout(1, 0));
		add(body, BorderLayout.CENTER);
	
		
		JPanel left = new JPanel(new BorderLayout());
		body.add(left);
		
		left.add(new JLabel("Independent variables (regressors)"), BorderLayout.NORTH);
		JCheckList<Wrapper> regressorCheckList = new JCheckList<>();
		List<Wrapper> newAttList = Util.newList();
		for (int i = 0; i < attList.size(); i++) {
			Attribute att = attList.get(i);
			Wrapper wrapper = new Wrapper(att) {

				/**
				 * Default serial version UID.
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public String toString() {
					// TODO Auto-generated method stub
					return att.getName();
				}
				
			};
			newAttList.add(wrapper);
		}
		regressorCheckList.setListData(newAttList);
		left.add(new JScrollPane(regressorCheckList), BorderLayout.CENTER);

		
		JPanel right = new JPanel(new BorderLayout());
		body.add(right);
		
		right.add(new JLabel("Dependent variables (responsors)"), BorderLayout.NORTH);
		JRadioList<Attribute> responsorRadioList = new JRadioList<>(attList.getList(), "");
		right.add(new JScrollPane(responsorRadioList), BorderLayout.CENTER);
		
		
		JPanel footer = new JPanel();
		add(footer, BorderLayout.SOUTH);
		
		JButton ok = new JButton("OK");
		ok.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				selectedRegressors.clear();
				selectedResponsor = null;
				
				List<Wrapper> wrappers = regressorCheckList.getCheckedItemList();
				for (Wrapper wrapper : wrappers) {
					selectedRegressors.add((Attribute)wrapper.getObject());
				}
				
				selectedResponsor = responsorRadioList.getSelectedItem();
				
				if (selectedRegressors.size() == 0 || selectedResponsor == null) {
					JOptionPane.showMessageDialog(
							ok, "You do not select any regressors nor any responsor", "No regressors-responsors", JOptionPane.ERROR_MESSAGE);
				}
				dispose();
			}
		});
		footer.add(ok);
	
		JButton cancel = new JButton("Cancel");
		cancel.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				selectedRegressors.clear();
				selectedResponsor = null;
				dispose();
			}
		});
		footer.add(cancel);

//		regressorCheckList.addListSelectionListener(new ListSelectionListener() {
//			
//			@Override
//			public void valueChanged(ListSelectionEvent e) {
//				// TODO Auto-generated method stub
//				
//				List<Wrapper> wrappers = regressorCheckList.getCheckedItemList();
//				List<Attribute> possibleResponsors = Util.newList();
//				for (int i = 0; i < attList.size(); i++) {
//					Attribute att = attList.get(i);
//					
//					boolean found = false;
//					for (Wrapper wraper : wrappers) {
//						Attribute checkedAtt = (Attribute)wraper.getObject();
//						if (checkedAtt.getName().equals(att.getName())) {
//							found = true;
//							break;
//						}
//					}
//					
//					if (!found) possibleResponsors.add(att);
//				}
//				
//				responsorRadioList.setListData(possibleResponsors);
//				
//			}
//		});

		
		setVisible(true);
	}


	/**
	 * Getting selected regressors.
	 * @return selected regressors.
	 */
	public List<Attribute> getSelectedRegressors() {
		return selectedRegressors;
	}
	
	
	/**
	 * Getting selected responsor.
	 * @return selected responsor.
	 */
	public Attribute getSelectedResponsor() {
		return selectedResponsor;
	}


	/**
	 * Getting regression indices like x1, x2,..., xn-1, z.
	 * @return regression indices like x1, x2,..., xn-1, z.
	 */
	public String getIndices() {
		if (selectedRegressors.size() == 0 || selectedResponsor == null)
			return null;
		else {
			StringBuffer buffer = new StringBuffer();
			for (int i = 0; i < selectedRegressors.size(); i++) {
				if (i > 0) buffer.append(", ");
				
				buffer.append(selectedRegressors.get(i).getIndex() + 1); //Index starts with 1.
			}
			
			buffer.append(", " + (selectedResponsor.getIndex() + 1)); //Index starts with 1.
			
			return buffer.toString();
		}
	}
}
