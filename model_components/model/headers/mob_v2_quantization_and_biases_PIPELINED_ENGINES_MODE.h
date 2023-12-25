#include "../../basic_defs/basic_defs_glue.h"
#if FIRST_PART_IMPLEMENTATION ==PIPELINED_ENGINES_MODE && MODEL_ID == MOB_V2
#ifndef BIAS_QUANT
#define BIAS_QUANT
const static biases_dt first_conv_layer_fused_zero_points[] ={ 31007, 13592, -8042, -1073741824, 125929, 87752, -51267, -1073741824, 14523, 29031, -15689, -1073741824, -1073741824, -1073741824, 81404, 46251, 93790, 35504, -71116, -1073741824, 107965, 130903, 20047, -3802, 120239, 37095, 130019, 12537, 1073741760, 118402, -11932, 380857};
const static fused_scales_dt first_conv_layer_fused_scales[] = { 0.0031877710814403955, 0.009079871437724537, 0.0065723387021488, 4.8117013316840675e-08, 0.00117021074183343, 0.001678376525017431, 0.005792371537079205, 1.006464498278849e-07, 0.007951715654955458, 0.0033724268991451724, 0.0070193878359384515, 1.1824302005919519e-08, 7.733574087469719e-09, 5.318358112728309e-08, 0.0015553415126806842, 0.0028517908614139114, 4.866921892069902e-06, 0.0031141177431525005, 0.002763344084324424, 5.179176048748312e-08, 0.0010127025902021592, 0.0011379550040666183, 0.004574250921363929, 0.0030025688878035518, 0.001115299262168708, 0.003493248002414, 5.3908969081128995e-05, 0.009524984662245795, 3.467612408650272e-09, 0.0009283637219314673, 0.00947134544687857, 0.0005300325008044454};
const static fused_scales_log_2_shifts_dt first_conv_layer_fused_scales_log_2_shifts[] ={ 7, 5, 6, 23, 8, 8, 6, 22, 5, 7, 6, 25, 25, 23, 8, 7, 16, 7, 7, 23, 8, 8, 6, 7, 8, 7, 13, 5, 27, 9, 5, 9};
const static layer_0_relu_6_fused_scales_dt first_conv_layer_relu_6_fused_scales[] ={ 255};
const static int layers_fused_parameters_offsets[] = { 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 224, 416, 448, 448, 640, 832, 864, 864, 1056, 1056, 1248, 1312, 1696, 2080, 2144, 2144, 2528, 2912, 2976, 2976, 3360, 3744, 3808, 3808, 4192, 4576, 4672, 5248, 5824, 5920, 5920, 6496, 7072, 7168, 7168, 7744, 7744, 8320, 8480, 9440, 10400, 10560, 10560, 11520, 12480, 12640, 12640, 13600, 14560, 14880, 16160, 16160, 16160, 16160, 16160};
const static int pipe_layers_fused_parameters_offsets[] = { 
0, 0, 32, 64, 80, 176, 176, 272, 296, 440, 584, 608, 608, 752, 752, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896};
const static biases_dt pipe_fused_zero_points[] = 
{ 6242, 15863, 694, -124174, 6092, 7244, 1504, 2964, 10913, 4795, 1320, 62319, -80454, 14440, 12319, 9266, -100040, 7700, 6793, 8466, -23736, 6726, -119, 76636, 8536, 7834, -36478, 17929, 29090, 4691, 1815, -8542, 24122, -44845, 977, -5325, -546, 14630, 13409, -475, 3020, -1425, 21989, 3813, 19865, -6293, 5061, 2965, 3768, -931, -362, 3405, 1063, 18980, -340, 5635, 13467, -112, 6612, 9390, 293, 11901, 10827, -995, -518, 18358, 1285, 403, -1498, 466, 1500, 1590, 6241, 15360, -6587, 11508, 13004, 240, 165, 4986, 2082, 18417, 6400, 2866, 156, 8722, -1973, -977, -348, -128, 17374, -1148, 20592, 609, 814, 450, 594, -158, 6002, 123, -541, 10763, -1918, 5416, 5219, 224, 2237, 1156, 613, -1294, 2869, 5040, 1268, -156, 10732, 274, 4815, -1703, -1954, 5164, 16429, 20377, -320, 9027, -1046, 4388, 8904, -229, 5130, 929, -1726, 46, 3721, 750, 1636, 206, 9667, 9887, -7981, -165, 554, 12016, 348, 20111, 777, 85467, 70418, 1832, 86743, -142, 59401, 952, 11062, 69257, 16815, 877, 69939, 1135, 17724, -10098, 78912, 96, -57710, -64131, 76878, 66813, -29439, -47847, 430, 15, -7825, 145, -4790, 15613, 674, -1139, -69339, 3293, 135, 13445, 5707, 14462, -31656, -61353, -84949, -74180, -14531, 68501, 82, -14586, 79545, 4681, -14128, 62922, -60, 74994, -20309, -36, 90500, -517, 4064, -63958, 51569, -59941, -43888, -68138, 62, -7083, -331, 79672, 29830, -54976, 266, 89205, -53988, -6593, 65, -46, -62625, 2422, 68138, 4367, 170, 20814, 439, -30339, -15876, -20912, 8399, 8254, -62316, -57330, -1516, -76, 13444, -62621, -13266, -598, -29768, 144, -39410, 60713, -53410, 76581, -51785, 46666, 28410, -49569, 14137, -9405, -28617, -67448, -104692, 53651, -4404, 43323, -73631, -65210, 84029, -44820, 24577, -41178, 98621, 13299, 1632, 1030, 2226, 222, -1194, 6876, 2002, -6697, 519, 154, 2471, -1126, 10527, 1235, 11923, 18994, 6215, 4483, 2219, -4537, 14249, 3875, 11245, 6357, 2006, -5631, 2538, 1139, 6020, -2993, 7323, 12809, -3561, 1955, 7157, 7360, -4347, 582, 1675, 9577, 65839, -245, 449, 54065, -7753, 12367, 181, -1596, 2771, 10750, 2155, 10606, -2801, -217, 6550, -4238, 3363, -4278, 6342, 36, 2893, 6627, 2516, -352, -510, 8696, 5061, 3531, 4679, 2818, 13423, 4079, 7667, -692, 1301, 20934, -1370, 5507, 17569, 4027, 4036, -4341, 3422, -2900, -4785, -7945, 2764, 5652, 5579, 3704, 11277, 5162, 7022, -4104, -4606, 4016, 11466, 1115, 5345, 3135, -315, 5570, -1249, 5415, -1244, 1926, -6357, 8727, 9272, -3683, 11687, -3923, -2535, -1601, 11339, 10800, 2742, 9757, 661, 5077, 1143, 736, 1776, 3494, 16093, 2815, 10439, 7491, -3303, 1355, -1073741824, 1115, -1593, 21580, 7955, 4647, 4124, 520, 4868, 7139, 3248, 2220, -2398, -3824, -15536, 7408, -3939, 5341, -875, 40, -575, 6465, 10580, -437, 1759, 9339, -238, 4794, 725, -14519, 12423, 9413, -4126, 15365, 120, 12466, 23143, 3239, 3555, 16551, -27935, -7483, 1764, 48181, 146, 7296, 6876, 46002, -14969, 146, -6041, 62840, 41753, 14715, -506, -24597, 15122, 5221, 23377, 550, -1140, -1312, -89, -2751, 827, -57, -7757, -1136, 1205, 17998, 138, 2311, 1137, 2228, 5617, -116, -8310, -2017, -878, -2492, 79, -159, 8302, 18837, 244, 14681, -141, 11261, 20082, -48, -40659, -61, -1016, 268, -17862, 2338, -375, -6891, 6012, 28570, 713, 2376, 21573, -1707, -12184, -5888, -19939, 11681, -25, 4125, -9363, -17511, -1264, -1097, -4425, -3801, -2798, -131, 7420, -1350, 66859, -1259, 631, 27030, 1792, 16804, -7430, 4809, -2931, -182, 2353, 7483, 615, 1653, 401, -7258, 8186, -11539, -34, 506, 7163, 18815, 252, -689, 9215, -4984, 3240, -2333, -575, -30837, -13779, 35391, -4489, 48, -28789, -1171, 88559, 20348, 4812, 142693, -47050, 152087, 32770, -69488, -15219, -88253, -18771, -27470, 80965, -193464, -34276, 176800, -12028, -42092, -132367, -7820, -10187, -28019, 28203, 248692, -47110, 1215, -2242, 399, 13, -1494, 57, 1488, 175, -848, 238, 2963, 601, 688, -1136, -3743, -2761, 8364, -293, -472, 2322, 14337, -4548, 1841, 1108, -2239, -932, 2480, 1963, 126, 442, 10167, 2959, 586, 2038, 4678, -1430, -1898, -317, 3191, 3182, -2212, 2803, 8935, -2121, 119, 2071, 424, 99, -412, 1072, 4399, 1632, 44, -228, 1515, 16282, -35, 2578, -1986, -582, 2506, -5745, -348, 2634, 4273, 3896, 18, -175, -1040, 3437, 2552, -253, -1054, 1317, -540, 11148, 1432, -1451, -1881, 1301, -3579, 610, -1597, 156, 3138, -1977, 8106, 1715, 625, -2915, 6983, 890, -2980, -530, -1992, 188, 456, 2302, -2320, 4623, 5095, 1219, 1075, -2761, 12863, 766, -1854, -944, 3005, -397, 10075, 1115, 6678, 8551, 632, 24022, 2700, 8642, -1735, 11770, -2449, 1348, -2405, -264, 1093, 3784, -753, -1251, -1317, 11686, -2710, 492, 6489, 13516, -1109, 833, 3009, -1048, -2800, -2103, -1074, 1112, 3693, 2124, 1868, 85242, -77269, -9182, -57212, -60046, -15418, 72841, 73513, -58792, 3968, 5238, -78245, 80733, 83571, 79822, 1464, 74415, 81169, -14961, 1656, 97083, 70222, 78748, 72501, -50682, -8734, -4232, -60164, 72242, 661, -1272, -54027, -2780, -2283, 73550, 77815, 81163, 16470, -6544, -71611, -4618, 1305, 80219, 56361, 75945, 74106, -45353, 86138, 73248, 1495, -46132, -46349, 73497, 72998, -2239, -34680, 999, 79091, -62333, -25881, 116579, -67963, -42358, -3765, -17603, -62998, 73283, -52620, 38166, 45045, 78316, 75237, 68734, -89403, -7512, -45419, -79661, -43738, 85922, 71916, 73591, -12814, -8925, -7539, 81468, 15752, -45244, -65443, -50759, -837, -48376, 80623, -24679, 79108, -75712, -83141, -47801, 79378, -51605, 50188, 84139, 76250, -63504, -1900, 76940, -73840, -36606, -87529, -59324, -679, 70933, -34190, -85, 80726, 689, 14275, -8264, 79332, -17666, 83223, -57659, 80458, -63200, 73405, -7694, -7448, -67895, 129990, 3408, 107058, -34392, -2911, -6554, -46981, -72334, 9040, 75738, 67258, 74206, 71602, 67328, -13595, -3038, 75867};
const static fused_scales_dt pipe_fused_scales[] ={ 0.01227943692356348, 0.008529157377779484, 0.007617452181875706, 0.4444526433944702, 0.02211803011596203, 0.017149513587355614, 0.01917203888297081, 0.08821505308151245, 0.013272730633616447, 0.010300838388502598, 0.0075339931063354015, 0.010839304886758327, 0.26341864466667175, 0.15913476049900055, 0.009964020922780037, 0.012176190502941608, 0.049389347434043884, 0.008332681842148304, 0.019266678020358086, 0.2429400384426117, 0.0005993210943415761, 0.01625228300690651, 0.023253843188285828, 0.0015173305291682482, 0.01541254110634327, 0.016017649322748184, 0.0022118815686553717, 0.006452914793044329, 0.007673711981624365, 0.022324329242110252, 0.010421888902783394, 0.010369190014898777, 0.001814370872649008, 0.0014434224059244637, 0.0018220940104612883, 0.002661892487698548, 0.0024209855860756434, 0.001758831619335281, 0.0014268217984578124, 0.00210001217884445, 0.0021202126950240897, 0.0024863993537129966, 0.0014791055216353442, 0.001573649083242838, 0.0020271841430243745, 0.0016255559458885648, 0.0014883276027120639, 0.001995984934178474, 0.016057163252003768, 0.030799526660651565, 0.016993747418875303, 0.030901945095220637, 0.02841061560337564, 0.004127753094662654, 0.020445547971758814, 0.010060024613621685, 0.009232608675039389, 0.017612394696226944, 0.0138809032911157, 0.009547045878466512, 0.034252331114974145, 0.009639494212101791, 0.0056657622278808726, 0.02245505878610272, 0.00863640669683834, 0.005107063139153454, 0.034106633166973634, 0.07980879358660976, 0.013392562048049483, 0.01500104403790082, 0.11152001734357854, 0.04093046601516689, 0.008088580102824353, 0.0063974354902112925, 0.013204490705448896, 0.005991116161010626, 0.009715002873031515, 0.1538237311261446, 0.014481884510676007, 0.01724331284656472, 0.06720727141581484, 0.01099041122569083, 0.017760947896372326, 0.025576048670715293, 0.05604047899108982, 0.01232478621584563, 0.10606413523923586, 0.06980117393187536, 0.07828278144809354, 0.05759959193920726, 0.010184242982130519, 0.00821445053177187, 0.003892714437425899, 0.033049996170681335, 0.011679906193005109, 0.061902656211796976, 0.012937027335498454, 0.028950682914652814, 0.015331997189400124, 0.021229419918237413, 0.013818134307075031, 0.00781382680665406, 0.010682633483479616, 0.008577063829118933, 0.013947913246541802, 0.06428790368070436, 0.05063693758849221, 0.036116272495601946, 0.03319615602509537, 0.043411447313328314, 0.05432932060683767, 0.024220940042676133, 0.017543065896103927, 0.010338925754884366, 0.009244743031023631, 0.07728639716340373, 0.023543481648738394, 0.01590633590778214, 0.08437410522547627, 0.0119451289443448, 0.0055258446288898935, 0.004875859888295909, 0.05071642646718684, 0.0072114379831743226, 0.013461830316082017, 0.01658816378738972, 0.008549023502281978, 0.15608608913874497, 0.016383278492579197, 0.009945633862818277, 0.11869191157456707, 0.11412978857997455, 0.025754595987325792, 0.08440234856763207, 0.053850189554153965, 0.08052969539625471, 0.005072873217555029, 0.009230376437795012, 0.00948723086044135, 0.04949783167212773, 0.05580625518891724, 0.00901927775671228, 0.1047202706798194, 0.004593901876463906, 0.0032054369803518057, 0.0033519272692501545, 0.0023778483737260103, 0.003082915907725692, 0.0024934893008321524, 0.013905667699873447, 0.0034428169019520283, 0.007024806458503008, 0.009802599437534809, 0.004729829728603363, 0.004892684519290924, 0.004061829764395952, 0.0025295130908489227, 0.006432210095226765, 0.010205947794020176, 0.002490393351763487, 0.004813730251044035, 0.011178375221788883, 0.002119585173204541, 0.002820871537551284, 0.003901825984939933, 0.0036587808281183243, 0.0015593132702633739, 0.003284243168309331, 0.00736596854403615, 0.011031176894903183, 0.022072067484259605, 0.024873968213796616, 0.009291283786296844, 0.0007185608264990151, 0.005276425741612911, 0.00382786151021719, 0.0041527580469846725, 0.0063948784954845905, 0.008591683581471443, 0.0033753197640180588, 0.001900046830996871, 0.0034367144107818604, 0.000808368728030473, 0.0023919984232634306, 0.002667119028046727, 0.0027590689714998007, 0.008250178769230843, 0.004897130653262138, 0.017230020835995674, 0.0020232086535543203, 0.00402881670743227, 0.0014511995250359178, 0.003281265264376998, 0.003111607162281871, 0.008829775266349316, 0.002518030581995845, 0.005018643103539944, 0.012228845618665218, 0.0030844907741993666, 0.004476251546293497, 0.005861656740307808, 0.0027036708779633045, 0.002485719509422779, 0.004905513022094965, 0.0017740164184942842, 0.003643086878582835, 0.0024230554699897766, 0.0028623556718230247, 0.005594132002443075, 0.004030819516628981, 0.0038504041731357574, 0.0021292001474648714, 0.0069359526969492435, 0.0034070112742483616, 0.0067761484533548355, 0.0050639864057302475, 0.01505084428936243, 0.01159717608243227, 0.0033422578126192093, 0.00974985584616661, 0.004093416966497898, 0.003348316764459014, 0.009600920602679253, 0.0006364004220813513, 0.005100928712636232, 0.007556755095720291, 0.0010364657500758767, 0.000826531439088285, 0.0051163979806005955, 0.0018552708206698298, 0.003547977888956666, 0.003932348918169737, 0.00916367582976818, 0.008617568761110306, 0.018906382843852043, 0.003480181796476245, 0.0017594093224033713, 0.012727069668471813, 0.0012064621550962329, 0.016413062810897827, 0.0008184901582643887, 0.0009966171607505097, 0.0011612942175104567, 0.0009452622382225594, 0.0007264947089391251, 0.0006646358119849953, 0.0009114582492111231, 0.0008675278170899735, 0.0009958068086599323, 0.0008486064792223553, 0.0016655723429279024, 0.001038343902625882, 0.0009683104191883198, 0.0013456041443657284, 0.0022171141170827823, 0.0008085081104753669, 0.0012274285890637945, 0.0011446962785029876, 0.0009888744455524836, 0.0014753650686907854, 0.001632101551414187, 0.0012043010162348113, 0.0006914828972578487, 0.0009952835673461067, 0.012542117948331342, 0.020003087946627, 0.010430894168915709, 0.014632816320360571, 0.021021884140334946, 0.007811721852226115, 0.021156348048042707, 0.02079470482837897, 0.027225350440751247, 0.017483729645462232, 0.01629044299564362, 0.02310645769636097, 0.009313001688363872, 0.02091648713087215, 0.00703576437038151, 0.0038755021904646023, 0.02713620957768959, 0.013259822878164863, 0.011078930430045795, 0.021379887503275498, 0.006770928424891538, 0.01982874445047375, 0.008475743348724718, 0.010365023450222987, 0.022944285156681368, 0.011732394560714347, 0.02443203502785034, 0.014934985290589413, 0.013790606617711554, 0.03231431844185453, 0.010280002661523154, 0.008996455877856059, 0.022091440897113557, 0.036295080701821594, 0.01011795407291875, 0.010883640686628074, 0.03203690554445298, 0.041334281232899804, 0.029187663703311474, 0.008648038312138165, 0.0014612364949619383, 0.02682779555149238, 0.02325637911959015, 0.0036064233328706166, 0.014983412307669522, 0.007470726506579966, 0.01577771751753148, 0.01913492867816315, 0.013568003846651818, 0.00725213247658098, 0.014794248122477012, 0.011017547185608448, 0.010944174086867205, 0.05450422399488387, 0.01156448390269728, 0.020267084439026858, 0.02207832138091474, 0.022848547424419944, 0.013432938739118244, 0.013373663405181275, 0.015748068717232888, 0.018839073048327333, 0.009733411077667658, 0.016383924373003234, 0.01898756495197069, 0.006645807607810088, 0.014929377061103203, 0.013609749381223342, 0.014679513960163962, 0.021966977688730464, 0.006169611699114055, 0.015007816567265419, 0.009519741690686276, 0.03174413162039095, 0.02654095940348775, 0.002814325120146991, 0.015217173644385119, 0.006350450751163092, 0.003378392692272414, 0.01819893774046362, 0.022864266202087886, 0.01626971421942725, 0.014979463686591353, 0.01902640097627409, 0.028482787338649958, 0.013250748471905147, 0.008662286005787887, 0.018642302572005297, 0.01120132283902887, 0.010285897388704134, 0.009413709339185339, 0.018129183716229357, 0.01428626544299072, 0.019990519159157505, 0.01425882252649745, 0.01268839357706128, 0.005845869675135682, 0.017743876974532884, 0.0067110474377967124, 0.014536707477095559, 0.02029174253854921, 0.011449038691701356, 0.018859546500173234, 0.014009717976449542, 0.02282342766280913, 0.018672432332164538, 0.013498226071313329, 0.005480172438868758, 0.01426817452378785, 0.028100842862645983, 0.0058509583492168706, 0.03738282189777598, 0.0130620147754768, 0.020100439269532023, 0.008437682946418817, 0.008120150722863468, 0.01541757803965428, 0.008863693142988741, 0.013798382135458715, 0.03212113289782714, 0.014418415102480113, 0.015039516869191843, 0.02779978574098389, 0.021242016796554804, 0.005732284101378734, 0.01937405776443463, 0.007616431366586387, 0.016182329453634563, 0.011363700231982453, 0.011254111894845143, 7.308839136713574e-08, 0.026459166538297118, 0.015268666038354858, 0.003151489486903109, 0.012105627577020113, 0.019154299188106388, 0.013084432848981498, 0.03465892057143005, 0.009776741998446506, 0.007142188615102235, 0.02418608640577694, 0.01833369111475418, 0.018857986349514154, 0.024069671852839647, 0.0070394366048276424, 0.022089330479502678, 0.008777466602623463, 0.015574585646390915, 0.006802508141845465, 0.01549216266721487, 0.013447986915707588, 0.030700847506523132, 0.007225424982607365, 0.0062903449870646, 0.00662081316113472, 0.017232311889529228, 0.012759225443005562, 0.005416216794401407, 0.019522257149219513, 0.015233788639307022, 0.007913371548056602, 0.006144073326140642, 0.009066748432815075, 0.027088167145848274, 0.012444999068975449, 0.010643517598509789, 0.01110080722719431, 0.0069129131734371185, 0.008257873356342316, 0.01515552680939436, 0.0034941257908940315, 0.009784912690520287, 0.006370580289512873, 0.0027230503037571907, 0.012078559026122093, 0.01688474416732788, 0.036313991993665695, 0.0024332115426659584, 0.009201127104461193, 0.019168458878993988, 0.018154164776206017, 0.0021015082020312548, 0.003761992324143648, 0.007106368895620108, 0.055077411234378815, 0.002261873334646225, 0.009161095134913921, 0.024882545694708824, 0.007534428499639034, 0.010616199113428593, 0.00406441418454051, 0.004412561189383268, 0.009871629998087883, 0.009337224997580051, 0.005960152018815279, 0.008639074862003326, 0.008939729072153568, 0.009883051738142967, 0.006341093685477972, 0.018893690779805183, 0.007996720261871815, 0.007602817844599485, 0.005136880557984114, 0.005483683664351702, 0.014956789091229439, 0.004857328720390797, 0.005817817058414221, 0.005998630076646805, 0.014422481879591942, 0.011783558875322342, 0.009179728105664253, 0.007739649619907141, 0.011996624059975147, 0.006422040518373251, 0.025144213810563087, 0.005568590946495533, 0.010277851484715939, 0.0095655657351017, 0.002844281494617462, 0.029740262776613235, 0.017946399748325348, 0.008007137104868889, 0.03846782445907593, 0.005628363694995642, 0.006892933044582605, 0.004586154595017433, 0.010883853770792484, 0.005990420002490282, 0.08454736322164536, 0.008481877855956554, 0.007783151231706142, 0.008011038415133953, 0.008108935318887234, 0.005826265085488558, 0.00922353658825159, 0.006371203809976578, 0.012595906853675842, 0.026163632050156593, 0.011258273385465145, 0.005425994284451008, 0.010518096387386322, 0.016105132177472115, 0.025947190821170807, 0.007779090665280819, 0.004592962563037872, 0.009774171747267246, 0.008916350081562996, 0.010954148136079311, 0.011408299207687378, 0.0054246606305241585, 0.002659799763932824, 0.012335354462265968, 0.006394801661372185, 0.0060151065699756145, 0.009199910797178745, 0.011740820482373238, 0.005170003976672888, 0.004511136095970869, 0.004703223705291748, 0.016903674229979515, 0.0051640975289046764, 0.016333628445863724, 0.008464807644486427, 0.003126543015241623, 0.007266692817211151, 0.015330641530454159, 0.013377333991229534, 0.00522882305085659, 0.019573448225855827, 0.00841742567718029, 0.011868496425449848, 0.006961323320865631, 0.020966343581676483, 0.008624142035841942, 0.2715984880924225, 0.017012132331728935, 0.0055791838094592094, 0.03723641857504845, 0.005854795221239328, 0.0023534544743597507, 0.008641524240374565, 0.003030494786798954, 0.006834725849330425, 0.0070435707457363605, 0.0023116657976061106, 0.008410010486841202, 0.002325530629605055, 0.009738001972436905, 0.0007231974427570372, 0.0003834022031274489, 0.001056895304276732, 0.00036054245642971113, 0.0014681040761537053, 0.0015907378972260413, 0.0007713261316815554, 0.000625247859393609, 0.0011162241303003617, 0.0016010521279239314, 0.0002864928111727835, 0.0004074901990303611, 0.001210338299553819, 0.0005036061926576671, 0.000351607246315881, 0.0013446849852979147, 0.0005782452622321541, 0.0013265868609564735, 0.0004383902031579743, 0.0006090785598575265, 0.0007785581758572768, 0.0004279172560850594, 0.0009845730794840817, 0.0006747427205585951, 0.017450981387724124, 0.024429093971930365, 0.02905693616701145, 0.027562514266325073, 0.034244596186339425, 0.015757903285410484, 0.03952904723970909, 0.022609680596181188, 0.0410781080688713, 0.07231865257839777, 0.03836153656022115, 0.021976176475746987, 0.03222709067056399, 0.021414891607164418, 0.011092371849834226, 0.006509405838930754, 0.018548146565056503, 0.01143116104633469, 0.019676740625372384, 0.005210113544432688, 0.020435937279520793, 0.010822839976563758, 0.013348368904767706, 0.011961858282604801, 0.04939538762416656, 0.0067138466980499545, 0.014559181600810685, 0.03783253635048456, 0.024710335570368648, 0.00408433767386479, 0.027090656159722896, 0.035562190827271224, 0.0120549466032613, 0.021904412346560556, 0.03949276573235121, 0.012207307388463765, 0.012323903521037172, 0.060917902441213, 0.02021345613263013, 0.01581075440490629, 0.014976708228822555, 0.007413313344121964, 0.01892961803661047, 0.034317328661622015, 0.022015445330811526, 0.015736689243308857, 0.030798000660120308, 0.01655163237178059, 0.03494881378044572, 0.009760834840766645, 0.051951511447278685, 0.050156144525962405, 0.022411837805274698, 0.018918713249135084, 0.005407155028944603, 0.018784577947894975, 0.016085016648856497, 0.023172708487736848, 0.017518178554634542, 0.01706310914847454, 0.012433091805974955, 0.03058682374346969, 0.02635815075528252, 0.01830165890800941, 0.014977140353267967, 0.03545581558774279, 0.01771462450629732, 0.045805075012085826, 0.047134686828070975, 0.024707960701567815, 0.020900503740018884, 0.01754060302935637, 0.01372741951108127, 0.022400099029723997, 0.007788146197970441, 0.020163662565932952, 0.02756600273313649, 0.016972147557931732, 0.016212887961287073, 0.01697945372922726, 0.04253921644298453, 0.01061242200116601, 0.040541920310235964, 0.008887193598027232, 0.01892272099154057, 0.015204689581539409, 0.031342695339210414, 0.05047874418616769, 0.012023545560228065, 0.00793654826328854, 0.017756546629665332, 0.02053270894664056, 0.0421974871056496, 0.011551986430483078, 0.026326536676108132, 0.04766121533439712, 0.030572947343911718, 0.03701940887094313, 0.010827047440351574, 0.017301770148199423, 0.030944051460073667, 0.01855233829322026, 0.021804960781619667, 0.007408265234879419, 0.0536401182114933, 0.017401894713677143, 0.016782542971438327, 0.024361261327894342, 0.01594690144516117, 0.00714767301172544, 0.01597155795763465, 0.01181284434359719, 0.007601444834116585, 0.035634269668938914, 0.0032276769851378563, 0.023152876759687486, 0.006127928315213688, 0.011436821029266747, 0.004771026387981657, 0.011525106353286291, 0.03924639185597041, 0.027860290374355293, 0.04125297684864639, 0.03406425867198627, 0.012447744092842479, 0.009802661339957663, 0.022001537460005424, 0.048890869624606134, 0.0063613348269342795, 0.020259331528935717, 0.050446294908485366, 0.011268267076471243, 0.003287447845315189, 0.027416737024143878, 0.033401314409363565, 0.030454978580747003, 0.04519597563414011, 0.016093355319177056, 0.016631203791367313, 0.03261707511473583, 0.014145213644702205, 0.014014392507976203, 0.018882510419897518, 0.01905892613256122, 0.0046935658901929855, 0.006195747293531895, 0.0037495880387723446, 0.003504781052470207, 0.002814119216054678, 0.004188052378594875, 0.0030906067695468664, 0.004405273124575615, 0.002514579566195607, 0.001680337474681437, 0.0016711604548618197, 0.005763590801507235, 0.0030483955051749945, 0.007102947682142258, 0.0056802984327077866, 0.010528245940804482, 0.00442883325740695, 0.0036251237615942955, 0.002567532006651163, 0.014630206860601902, 0.008858220651745796, 0.0042134420946240425, 0.004926545545458794, 0.005895433947443962, 0.0019173083128407598, 0.007597106043249369, 0.006647769827395678, 0.0025730375200510025, 0.003146582283079624, 0.01316578034311533, 0.0045385099947452545, 0.004040898289531469, 0.008851618506014347, 0.0024402348790317774, 0.003545469371601939, 0.005467960145324469, 0.0036931366194039583, 0.0015188182005658746, 0.0046423268504440784, 0.004180279094725847, 0.0062880730256438255, 0.013219447806477547, 0.004245379939675331, 0.0015031184302642941, 0.002754224231466651, 0.0034557958133518696, 0.00216610892675817, 0.0032791919074952602, 0.0030304393731057644, 0.00671321852132678, 0.003541705897077918, 0.0018225484527647495, 0.003244881983846426, 0.003963220398873091, 0.00845322571694851, 0.0037428895011544228, 0.004204314202070236, 0.0049986629746854305, 0.004144108388572931, 0.003209308022633195, 0.00685068080201745, 0.0037050864193588495, 0.0019461465999484062, 0.003582234727218747, 0.008664212189614773, 0.0029027718119323254, 0.004188201855868101, 0.0018707596464082599, 0.0016471361741423607, 0.0020640192087739706, 0.0051958151161670685, 0.005090172868221998, 0.0031976187601685524, 0.002143386285752058, 0.004373637959361076, 0.0025666530709713697, 0.003710808465257287, 0.0039827474392950535, 0.0032759192399680614, 0.006383797153830528, 0.00356667791493237, 0.0034291022457182407, 0.0030057404655963182, 0.006039188709110022, 0.007182867266237736, 0.002672134665772319, 0.002077767625451088, 0.0033635529689490795, 0.004991153720766306, 0.006319283973425627, 0.0023360266350209713, 0.004817930515855551, 0.001544162747450173, 0.007528702728450298, 0.006040060892701149, 0.004956397693604231, 0.0033521850127726793, 0.005272493697702885, 0.004087178036570549, 0.0034869161900132895, 0.0027537876740098, 0.003855958115309477, 0.003825648222118616, 0.005793274845927954, 0.0019531925208866596, 0.004065794870257378, 0.0037056724540889263, 0.003197867888957262, 0.0037817773409187794, 0.011310736648738384, 0.004047490190714598, 0.0039895824156701565, 0.0058002048172056675, 0.0025557512417435646, 0.019739432260394096, 0.0021948544308543205, 0.010967169888317585, 0.005862786900252104, 0.008653971366584301, 0.007917783223092556, 0.0031709973700344563, 0.006264329422265291, 0.0025847535580396652, 0.0014881002716720104, 0.006069798953831196, 0.010557745583355427, 0.0032294795382767916, 0.0018168595852330327, 0.01119711808860302, 0.005161167122423649, 0.0018700018990784883, 0.004551983438432217, 0.015675274655222893, 0.0029040530789643526, 0.0028160761576145887, 0.0014470487367361784, 0.0059539624489843845, 0.006560539361089468, 0.005102778784930706, 0.003873251611366868, 0.004041914828121662, 0.006131600122898817, 0.004049794748425484, 0.003558075986802578};
const static fused_scales_log_2_shifts_dt pipe_fused_scales_log_2_shifts[] ={ };
const static relu_6_fused_scales_dt pipe_relu_6_fused_scales[] ={ 0, 0, 255, 16, 255, 0, 255, 20, 255, 255, 14, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static biases_dt seml_fused_zero_points_buffer[1280];
static fused_scales_dt seml_fused_scales_buffer[1280];
const static fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[] ={ };
const static relu_6_fused_scales_dt relu_6_fused_scales[] ={ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 255, 255, 27, 0, 255, 255, 21, 0, 255, 0, 255, 29, 255, 255, 31, 0, 255, 255, 39, 0, 255, 255, 33, 0, 255, 255, 36, 255, 255, 42, 0, 255, 255, 32, 0, 255, 0, 255, 44, 255, 255, 59, 0, 255, 255, 27, 0, 255, 255, 41, 255, 0, 0, 0, 0};
#endif
#endif
