
;
xPlaceholder*
dtype0*
shape:���������
;
yPlaceholder*
dtype0*
shape:���������
K
truncated_normal/shapeConst*
valueB"      *
dtype0
B
truncated_normal/meanConst*
valueB
 *    *
dtype0
D
truncated_normal/stddevConst*
valueB
 *���=*
dtype0
z
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
seed2 *
dtype0*

seed *
T0
_
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0
M
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0
\
Variable
VariableV2*
	container *
shared_name *
dtype0*
shape
:
�
Variable/AssignAssignVariabletruncated_normal*
_class
loc:@Variable*
T0*
validate_shape(*
use_locking(
I
Variable/readIdentityVariable*
_class
loc:@Variable*
T0
6
ConstConst*
valueB*    *
dtype0
Z

Variable_1
VariableV2*
	container *
shared_name *
dtype0*
shape:

Variable_1/AssignAssign
Variable_1Const*
_class
loc:@Variable_1*
T0*
validate_shape(*
use_locking(
O
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0
M
truncated_normal_1/shapeConst*
valueB"       *
dtype0
D
truncated_normal_1/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_1/stddevConst*
valueB
 *���=*
dtype0
~
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
seed2 *
dtype0*

seed *
T0
e
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0
S
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0
^

Variable_2
VariableV2*
	container *
shared_name *
dtype0*
shape
: 
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
_class
loc:@Variable_2*
T0*
validate_shape(*
use_locking(
O
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0
8
Const_1Const*
valueB *    *
dtype0
Z

Variable_3
VariableV2*
	container *
shared_name *
dtype0*
shape: 
�
Variable_3/AssignAssign
Variable_3Const_1*
_class
loc:@Variable_3*
T0*
validate_shape(*
use_locking(
O
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0
M
truncated_normal_2/shapeConst*
valueB"       *
dtype0
D
truncated_normal_2/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_2/stddevConst*
valueB
 *���=*
dtype0
~
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
seed2 *
dtype0*

seed *
T0
e
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0
S
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0
^

Variable_4
VariableV2*
	container *
shared_name *
dtype0*
shape
: 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
_class
loc:@Variable_4*
T0*
validate_shape(*
use_locking(
O
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0
8
Const_2Const*
valueB*    *
dtype0
Z

Variable_5
VariableV2*
	container *
shared_name *
dtype0*
shape:
�
Variable_5/AssignAssign
Variable_5Const_2*
_class
loc:@Variable_5*
T0*
validate_shape(*
use_locking(
O
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0
Q
MatMulMatMulxVariable/read*
transpose_a( *
transpose_b( *
T0
K
BiasAddBiasAddMatMulVariable_1/read*
data_formatNHWC*
T0

ReluReluBiasAdd*
T0
X
MatMul_1MatMulReluVariable_2/read*
transpose_a( *
transpose_b( *
T0
O
	BiasAdd_1BiasAddMatMul_1Variable_3/read*
data_formatNHWC*
T0
"
Relu_1Relu	BiasAdd_1*
T0
Z
MatMul_2MatMulRelu_1Variable_4/read*
transpose_a( *
transpose_b( *
T0
O
	BiasAdd_2BiasAddMatMul_2Variable_5/read*
data_formatNHWC*
T0
!
y_outRelu	BiasAdd_2*
T0
<
y_argout/dimensionConst*
value	B :*
dtype0
U
y_argoutArgMaxy_outy_argout/dimension*
output_type0	*

Tidx0*
T0
.
RankConst*
value	B :*
dtype0
.
ShapeShapey_out*
out_type0*
T0
0
Rank_1Const*
value	B :*
dtype0
0
Shape_1Shapey_out*
out_type0*
T0
/
Sub/yConst*
value	B :*
dtype0
"
SubSubRank_1Sub/y*
T0
6
Slice/beginPackSub*
N*

axis *
T0
8

Slice/sizeConst*
valueB:*
dtype0
F
SliceSliceShape_1Slice/begin
Slice/size*
Index0*
T0
F
concat/values_0Const*
valueB:
���������*
dtype0
5
concat/axisConst*
value	B : *
dtype0
U
concatConcatV2concat/values_0Sliceconcat/axis*
N*

Tidx0*
T0
8
ReshapeReshapey_outconcat*
T0*
Tshape0
0
Rank_2Const*
value	B :*
dtype0
,
Shape_2Shapey*
out_type0*
T0
1
Sub_1/yConst*
value	B :*
dtype0
&
Sub_1SubRank_2Sub_1/y*
T0
:
Slice_1/beginPackSub_1*
N*

axis *
T0
:
Slice_1/sizeConst*
valueB:*
dtype0
L
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
Index0*
T0
H
concat_1/values_0Const*
valueB:
���������*
dtype0
7
concat_1/axisConst*
value	B : *
dtype0
]
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
N*

Tidx0*
T0
8
	Reshape_1Reshapeyconcat_1*
T0*
Tshape0
[
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitsReshape	Reshape_1*
T0
1
Sub_2/yConst*
value	B :*
dtype0
$
Sub_2SubRankSub_2/y*
T0
;
Slice_2/beginConst*
valueB: *
dtype0
9
Slice_2/sizePackSub_2*
N*

axis *
T0
J
Slice_2SliceShapeSlice_2/beginSlice_2/size*
Index0*
T0
S
	Reshape_2ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
T0*
Tshape0
5
Const_3Const*
valueB: *
dtype0
E
lossSum	Reshape_2Const_3*
	keep_dims( *

Tidx0*
T0
8
gradients/ShapeConst*
valueB *
dtype0
<
gradients/ConstConst*
valueB
 *  �?*
dtype0
A
gradients/FillFillgradients/Shapegradients/Const*
T0
O
!gradients/loss_grad/Reshape/shapeConst*
valueB:*
dtype0
p
gradients/loss_grad/ReshapeReshapegradients/Fill!gradients/loss_grad/Reshape/shape*
T0*
Tshape0
F
gradients/loss_grad/ShapeShape	Reshape_2*
out_type0*
T0
s
gradients/loss_grad/TileTilegradients/loss_grad/Reshapegradients/loss_grad/Shape*

Tmultiples0*
T0
_
gradients/Reshape_2_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
T0
|
 gradients/Reshape_2_grad/ReshapeReshapegradients/loss_grad/Tilegradients/Reshape_2_grad/Shape*
T0*
Tshape0
K
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0
n
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_2_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0
E
gradients/Reshape_grad/ShapeShapey_out*
out_type0*
T0
�
gradients/Reshape_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_grad/Shape*
T0*
Tshape0
Y
gradients/y_out_grad/ReluGradReluGradgradients/Reshape_grad/Reshapey_out*
T0
r
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/y_out_grad/ReluGrad*
data_formatNHWC*
T0
x
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/y_out_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/y_out_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*0
_class&
$"loc:@gradients/y_out_grad/ReluGrad*
T0
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
T0
�
gradients/MatMul_2_grad/MatMulMatMul1gradients/BiasAdd_2_grad/tuple/control_dependencyVariable_4/read*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_2_grad/MatMul_1MatMulRelu_11gradients/BiasAdd_2_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
�
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul*
T0
�
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1*
T0
m
gradients/Relu_1_grad/ReluGradReluGrad0gradients/MatMul_2_grad/tuple/control_dependencyRelu_1*
T0
s
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*
T0
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
T0
�
gradients/MatMul_1_grad/MatMulMatMul1gradients/BiasAdd_1_grad/tuple/control_dependencyVariable_2/read*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu1gradients/BiasAdd_1_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0
i
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*
T0
o
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
data_formatNHWC*
T0
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*
T0
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
T0
�
gradients/MatMul_grad/MatMulMatMul/gradients/BiasAdd_grad/tuple/control_dependencyVariable/read*
transpose_a( *
transpose_b(*
T0
�
gradients/MatMul_grad/MatMul_1MatMulx/gradients/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
c
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@Variable*
dtype0
t
beta1_power
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable*
shape: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_class
loc:@Variable*
T0*
validate_shape(*
use_locking(
O
beta1_power/readIdentitybeta1_power*
_class
loc:@Variable*
T0
c
beta2_power/initial_valueConst*
valueB
 *w�?*
_class
loc:@Variable*
dtype0
t
beta2_power
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable*
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_class
loc:@Variable*
T0*
validate_shape(*
use_locking(
O
beta2_power/readIdentitybeta2_power*
_class
loc:@Variable*
T0
q
Variable/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable*
dtype0
~
Variable/Adam
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable*
shape
:
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
_class
loc:@Variable*
T0*
validate_shape(*
use_locking(
S
Variable/Adam/readIdentityVariable/Adam*
_class
loc:@Variable*
T0
s
!Variable/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable*
dtype0
�
Variable/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable*
shape
:
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
_class
loc:@Variable*
T0*
validate_shape(*
use_locking(
W
Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*
T0
q
!Variable_1/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_1*
dtype0
~
Variable_1/Adam
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable_1*
shape:
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
_class
loc:@Variable_1*
T0*
validate_shape(*
use_locking(
Y
Variable_1/Adam/readIdentityVariable_1/Adam*
_class
loc:@Variable_1*
T0
s
#Variable_1/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_1*
dtype0
�
Variable_1/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable_1*
shape:
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
_class
loc:@Variable_1*
T0*
validate_shape(*
use_locking(
]
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
T0
u
!Variable_2/Adam/Initializer/zerosConst*
valueB *    *
_class
loc:@Variable_2*
dtype0
�
Variable_2/Adam
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable_2*
shape
: 
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
_class
loc:@Variable_2*
T0*
validate_shape(*
use_locking(
Y
Variable_2/Adam/readIdentityVariable_2/Adam*
_class
loc:@Variable_2*
T0
w
#Variable_2/Adam_1/Initializer/zerosConst*
valueB *    *
_class
loc:@Variable_2*
dtype0
�
Variable_2/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable_2*
shape
: 
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
_class
loc:@Variable_2*
T0*
validate_shape(*
use_locking(
]
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_class
loc:@Variable_2*
T0
q
!Variable_3/Adam/Initializer/zerosConst*
valueB *    *
_class
loc:@Variable_3*
dtype0
~
Variable_3/Adam
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable_3*
shape: 
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
_class
loc:@Variable_3*
T0*
validate_shape(*
use_locking(
Y
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
T0
s
#Variable_3/Adam_1/Initializer/zerosConst*
valueB *    *
_class
loc:@Variable_3*
dtype0
�
Variable_3/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable_3*
shape: 
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
_class
loc:@Variable_3*
T0*
validate_shape(*
use_locking(
]
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
T0
u
!Variable_4/Adam/Initializer/zerosConst*
valueB *    *
_class
loc:@Variable_4*
dtype0
�
Variable_4/Adam
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable_4*
shape
: 
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
_class
loc:@Variable_4*
T0*
validate_shape(*
use_locking(
Y
Variable_4/Adam/readIdentityVariable_4/Adam*
_class
loc:@Variable_4*
T0
w
#Variable_4/Adam_1/Initializer/zerosConst*
valueB *    *
_class
loc:@Variable_4*
dtype0
�
Variable_4/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable_4*
shape
: 
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
_class
loc:@Variable_4*
T0*
validate_shape(*
use_locking(
]
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_class
loc:@Variable_4*
T0
q
!Variable_5/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_5*
dtype0
~
Variable_5/Adam
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable_5*
shape:
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
_class
loc:@Variable_5*
T0*
validate_shape(*
use_locking(
Y
Variable_5/Adam/readIdentityVariable_5/Adam*
_class
loc:@Variable_5*
T0
s
#Variable_5/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_5*
dtype0
�
Variable_5/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_class
loc:@Variable_5*
shape:
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
_class
loc:@Variable_5*
T0*
validate_shape(*
use_locking(
]
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_class
loc:@Variable_5*
T0
@
train/learning_rateConst*
valueB
 *o�:*
dtype0
8
train/beta1Const*
valueB
 *fff?*
dtype0
8
train/beta2Const*
valueB
 *w�?*
dtype0
:
train/epsilonConst*
valueB
 *w�+2*
dtype0
�
train/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_class
loc:@Variable*
use_locking( 
�
!train/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_class
loc:@Variable_1*
use_locking( 
�
!train/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_class
loc:@Variable_2*
use_locking( 
�
!train/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_class
loc:@Variable_3*
use_locking( 
�
!train/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon2gradients/MatMul_2_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_class
loc:@Variable_4*
use_locking( 
�
!train/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_class
loc:@Variable_5*
use_locking( 
�
	train/mulMulbeta1_power/readtrain/beta1 ^train/update_Variable/ApplyAdam"^train/update_Variable_1/ApplyAdam"^train/update_Variable_2/ApplyAdam"^train/update_Variable_3/ApplyAdam"^train/update_Variable_4/ApplyAdam"^train/update_Variable_5/ApplyAdam*
_class
loc:@Variable*
T0
}
train/AssignAssignbeta1_power	train/mul*
_class
loc:@Variable*
T0*
validate_shape(*
use_locking( 
�
train/mul_1Mulbeta2_power/readtrain/beta2 ^train/update_Variable/ApplyAdam"^train/update_Variable_1/ApplyAdam"^train/update_Variable_2/ApplyAdam"^train/update_Variable_3/ApplyAdam"^train/update_Variable_4/ApplyAdam"^train/update_Variable_5/ApplyAdam*
_class
loc:@Variable*
T0
�
train/Assign_1Assignbeta2_powertrain/mul_1*
_class
loc:@Variable*
T0*
validate_shape(*
use_locking( 
�
trainNoOp ^train/update_Variable/ApplyAdam"^train/update_Variable_1/ApplyAdam"^train/update_Variable_2/ApplyAdam"^train/update_Variable_3/ApplyAdam"^train/update_Variable_4/ApplyAdam"^train/update_Variable_5/ApplyAdam^train/Assign^train/Assign_1
�
init_all_vars_opNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^beta1_power/Assign^beta2_power/Assign^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign"