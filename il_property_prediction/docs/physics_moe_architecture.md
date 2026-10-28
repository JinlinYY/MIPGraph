# MIPGraph architecture

MIPGraph represents an ionic liquid as a cation–anion pair and predicts density,
viscosity, electrical conductivity, molar heat capacity, surface tension, and
thermal conductivity under specified temperature and pressure conditions.

## Molecular representation

A shared frozen Uni-Mol2-84M encoder provides CLS and atom-level features for
the cation and anion. Trainable projections map these features to a common
representation space. Chemistry-biased cross-ion attention describes atom-level
interactions between the ions.

The complete ion-pair representation combines cation and anion features,
cross-ion interaction features, global descriptors, and functional-group
descriptors. A two-layer, eight-head Transformer fuses these inputs.

## Conditions and property-specific decoding

Training-fitted temperature and pressure scales define the continuous condition
inputs. FiLM modulates the fused ion-pair representation. Each property has its
own router over four shared experts: packing, cohesion, transport, and thermal
response. Top-two routing combines the selected expert outputs.

Six independent prediction heads combine the condition-modulated representation,
condition features, and routed latent information. Each head has three hidden
layers with 512 units. The model returns standardized natural-log targets;
the fitted target scales convert them to physical-unit predictions.

## Sparse-label objective and evaluation

Only available training labels contribute to the masked objective. Effective
weights combine source reliability and experimental uncertainty. Property-wise
losses are normalized by their effective-weight sums and combined with the
configured property coefficients and auxiliary regularization terms.

Target statistics are fitted only to original experimental training observations
and are fixed across label-completion settings. Evaluation uses original labels.
For each property, log-space MAE is divided by the training-fitted target scale;
macro NMAE is the unweighted mean over the six properties.

The public model components are under `il_property_prediction/src/models/`.
Configuration files define representation dimensions, routing, loss coefficients,
and optimization settings for each experiment.
