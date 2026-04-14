Wannacry case study

Dates of attack : 12–13 May 2017                                                  
Organisations affected: 80+ NHS trusts, 595 GP practices                              
Estimated cost to NHS: £92 million                                                 
Appointments cancelled: ~19,000                                           

This notebook reconstructs the WannaCry attack using publicly available information from the NAO (2018), NHS Digital, 
and the NCSC post-incident review. The goal is not just to understand what happened, but to draw lessons for healthcare cybersecurity policy.

Based on literatures reported                                                      
1. https://www.nature.com/articles/s41746-019-0161-6
2. https://www.nao.org.uk/wp-content/uploads/2017/10/Investigation-WannaCry-cyber-attack-and-the-NHS.pdf

Sidebar slider — drag the Z-score threshold from 0.5 (very sensitive) to 3.5 (very strict). 

Every chart and table updates instantly in real time.                      
###Tab 1 – Scatter Plot — interactive Plotly chart. Hover over any dot to see the trust name,                      
% change, and whether it was flagged. The red dashed threshold line moves with the slider.                        
###Tab 2 – Model Performance — live confusion matrix, precision/recall/F1 bars,                            
and a precision–recall curve with a ⭐ star marking your current threshold position.                        
###Tab 3 – Trust List — three side-by-side tables showing exactly which named trusts are true positives,                     
false positives, and missed at the current threshold. Includes a CSV download button.
