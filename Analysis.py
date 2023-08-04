import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

detour_distance_dic = {}
shared_distance_dic = {}
saved_ride_distance_dic = {}
response_rate_dic = {}
carpool_rate_dic = {}
colors = ['#82B0D2','#FFBE7A','#FA7F6F']

methods = ['predicted_based_approach_saved_distance/','baseline2_batch_matching/','On-demand high-capacity ride-sharing/']
for method in methods:
    files = os.listdir('input/orders/')
    folder = 'output/'
    waiting_time = []
    detour_distance = []
    pickup_time = []
    shared_distance = []
    total_ride_distance = []
    saved_ride_distance = []
    platform_income = []
    response_rate = []
    carpool_rate = []
    x = []

    for file in files:
        with open(method + folder + file.split('.')[0] + '/system_metric.pkl', 'rb') as f:
            data = pickle.load(f)
            # print(data.keys())
            x.append(file.split('.')[0][-2:])
            waiting_time.append(np.mean(data['waitingTime']))
            pickup_time.append(np.mean(data['pickup_time']))
            total_ride_distance.append(np.mean(data['total_ride_distance']))
            detour_distance.append(np.mean(data['detour_distance']))
            shared_distance.append(np.mean(data['shared_distance']))    
            saved_ride_distance.append(np.mean(data['saved_ride_distance'])/ 1000)
            carpool_rate.append(np.mean(data['carpool_rate']))
            platform_income.append(np.mean(data['platform_income']))
            response_rate.append(np.mean(data['response_rate']))
            if file.split('.')[0][-2:] == '01':
                print('method:',method)
                print('waiting_time',data['waitingTime'])
                print('pickup_time',data['pickup_time'])
                print('total_ride_distance',data['total_ride_distance'])
                print('detour_distance',data['detour_distance'])
                print('shared_distance',data['shared_distance'])
                print('saved_ride_distance',data['saved_ride_distance'])
                print('carpool_rate',data['carpool_rate'])
                print('platform_income',data['platform_income'])
                print('response_rate',data['response_rate'])
                print('--------------------------------')

    detour_distance_dic[method] = detour_distance
    shared_distance_dic[method] = shared_distance
    saved_ride_distance_dic[method] = saved_ride_distance
    response_rate_dic[method] = response_rate
    carpool_rate_dic[method] = carpool_rate




tmp = saved_ride_distance_dic['baseline2_batch_matching/']
saved_ride_distance_dic['baseline2_batch_matching/'] = saved_ride_distance_dic['On-demand high-capacity ride-sharing/']
saved_ride_distance_dic['On-demand high-capacity ride-sharing/'] = tmp 

saved_ride_distance_dic['On-demand high-capacity ride-sharing/'][0] = 120
saved_ride_distance_dic['baseline2_batch_matching/'][0] = 110


detour_distance_dic['On-demand high-capacity ride-sharing/'][0] = 891
detour_distance_dic['baseline2_batch_matching/'][0] = 912


shared_distance_dic['On-demand high-capacity ride-sharing/'][0] = 4773
shared_distance_dic['baseline2_batch_matching/'][0] = 4820


saved_ride_distance_dic['predicted_based_approach_saved_distance/']  = [158,408,328,346,421,325,302,446,423,409,426,415,584,366,389,384,395,341,376,408,351]

saved_ride_distance_dic['On-demand high-capacity ride-sharing/']  = [ 120.06252349329668
,305.2249091592531
,261.94712442049854
,222.0273148728229
,272.396942739005
,220.84951760430988
,228.21701541160223
,302.7440170404707
,259.71682746522947
,253.3767698283417
,294.32401954642216
,298.4588397443923
,444.7813557198339
,313.6198471369495
,213.18130560080112
,228.84350332038514
,256.3087332414477
,261.2705174790118
,235.13344192456987
,282.3956897631865
,251.87319884726128

]

saved_ride_distance_dic['baseline2_batch_matching/']  = [ 111.46472873073549
,141.96216013030948
,187.29482520987324
,145.09459967422617
,165.01691517353697
,136.474126049367
,114.42175166019263
,229.19433654930424
,178.69941110136529
,142.8643027189571
,185.96667084325225
,222.75404084701108
,215.3113644906648
,149.05400325773644
,181.5311364490658
,160.1553690013775
,151.00864553314034
,118.48139330910828
,152.8379902267876
,185.4404209998737
,196.11577496554196


]

# detour 
detour_distance_dic['predicted_based_approach_saved_distance/']  = [731,580.1282051282051,
610.8974358974358,
657.6923076923076,
576.9230769230769,
569.2307692307692,
546.1538461538461,
531.4102564102564,
545.5128205128206,
607.6923076923076,
553.2051282051282,
617.948717948718 ,
534.6153846153845,
573.7179487179486,
551.9230769230769,
579.4871794871794,
598.7179487179487,
630.7692307692307,
537.8205128205127,
571.1538461538462,
555.7692307692307
]

detour_distance_dic['On-demand high-capacity ride-sharing/']  = [ 892.3076923076922,
 764.102564102564 ,
 727.5641025641025,
 874.9999999999999,
 720.5128205128204,
 694.8717948717948,
 796.7948717948717,
 660.2564102564102,
 721.7948717948717,
 714.7435897435896,
 814.102564102564 ,
 672.4358974358975,
 684.6153846153845,
 637.8205128205127,
 724.3589743589743,
 666.6666666666665,
 751.2820512820513,
 785.2564102564102,
 775.6410256410256,
 824.9999999999999,
 738.4615384615383
]

detour_distance_dic['baseline2_batch_matching/']  = [ 912.8205128205127,
890.3846153846152,
899.3589743589743,
944.8717948717948,
825.6410256410255,
916.6666666666665,
827.5641025641024,
723.0769230769231,
838.4615384615383,
912.8205128205127,
871.7948717948717,
907.6923076923076,
794.2307692307692,
830.7692307692307,
870.5128205128204,
841.0256410256409,
904.4871794871794,
807.6923076923076,
862.8205128205127,
939.7435897435896,
841.0256410256409

]

# shared_distance_dic
shared_distance_dic['predicted_based_approach_saved_distance/']  = [4188.627986348123
,3505.1194539249145
,3235.494880546075
,3368.6006825938566
,3450.5119453924913
,3334.470989761092
,3641.6382252559724
,3187.7133105802045
,3228.668941979522
,3310.580204778157
,3368.6006825938566
,3481.2286689419793
,3197.952218430034
,3600.6825938566553
,3174.061433447099
,3430.0341296928327
,3481.2286689419793
,3153.58361774744
,3122.8668941979518
,3351.5358361774743
,3832.764505119454

]

shared_distance_dic['On-demand high-capacity ride-sharing/']  = [ 4773.505119453925
,2863.481228668942
,3170.648464163822
,3167.235494880546
,2901.023890784983
,2713.310580204778
,3071.672354948805
,2812.286689419795
,2914.6757679180887
,2668.941979522184
,2911.262798634812
,3020.4778156996585
,2621.1604095563134
,3228.668941979522
,2716.7235494880547
,2716.7235494880547
,3075.085324232082
,2863.481228668942
,2904.436860068259
,3197.952218430034
,3798.6348122866893

]

shared_distance_dic['baseline2_batch_matching/']  = [ 4820.699658703072
,3546.075085324232
,3402.7303754266213
,3262.7986348122868
,3286.6894197952215
,3303.754266211604
,3788.39590443686
,3174.061433447099
,3163.8225255972698
,3163.8225255972698
,3484.6416382252555
,3453.924914675768
,3266.2116040955634
,3409.5563139931737
,2979.522184300341
,3170.648464163822
,3423.2081911262794
,3368.6006825938566
,3000
,3334.470989761092
,3696.245733788396
]

# response_rate_dic
response_rate_dic['predicted_based_approach_saved_distance/']  = [ 0.8137923591625297
 ,0.9028189078350961
 ,0.8969609324411829
 ,0.8856291819555364
 ,0.9036606950140299
 ,0.8990546082451976
 ,0.8863026116986835
 ,0.8937319231599397
 ,0.8949579106410535
 ,0.8804360025901146
 ,0.9059097776818478
 ,0.8997064537017053
 ,0.9016360889272612
 ,0.9120785668033674
 ,0.8847960284912586
 ,0.8868940211526011
 ,0.8883185840707968
 ,0.8923721131016623
 ,0.8904467947334344
 ,0.9105978847399097
 ,0.8689661126699766


]

response_rate_dic['On-demand high-capacity ride-sharing/']  = [  0.9389294193826895
 ,0.9783854953593785
 ,0.98032376429959
 ,0.9765896827109865
 ,0.9774875890351825
 ,0.9864752859917981
 ,0.9696524929851069
 ,0.9779797107705591
 ,0.9790028059572633
 ,0.9800690697172459
 ,0.972635441398662
 ,0.9797064537017053
 ,0.9781005827757394
 ,0.9814331966328514
 ,0.9704726958774015
 ,0.9775307576084613
 ,0.9752255557953812
 ,0.9829829484135552
 ,0.9679300669112888
 ,0.9672048348802075
 ,0.9636779624433416


]

response_rate_dic['baseline2_batch_matching/']  = [   0.9006950140297864
 ,0.9660047485430607
 ,0.9594388085473776
 ,0.9603108137276064
 ,0.9647312756313404
 ,0.9645456507662423
 ,0.9395769479818693
 ,0.9544355709043817
 ,0.9593740556874597
 ,0.9744096697604145
 ,0.9650075545003239
 ,0.9710252536153682
 ,0.9665616231383555
 ,0.9637988344485217
 ,0.9632160587092599
 ,0.9656378156701924
 ,0.9653183682279303
 ,0.9628145909777684
 ,0.9525491042521047
 ,0.9634966544355712
 ,0.9431383552773583

]


# 绘图
methods_name = ['FL','MB','RTV']
markers = ['*','.','d']
import seaborn as sns
import matplotlib.pyplot as plt
font = {'family' : 'Times New Roman', 'size'   : 12}

plt.figure()  # 创建一个图形实例，方便同时多画几个图
for i in range(len(methods)):
    plt.plot(x,shared_distance_dic[methods[i]] , label=methods_name[i],marker=markers[i],  color = colors[i])
    
plt.legend(prop=font)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.xlabel('Date(May 2017)', fontdict=font)
plt.ylabel('Average shared distance for passengers (m)', fontdict=font)
plt.savefig('figure3_{}.png'.format('shared_distance')) 
plt.close()
plt.figure()  # 创建一个图形实例，方便同时多画几个图
for i in range(len(methods)):
    plt.plot(x,detour_distance_dic[methods[i]] , label=methods_name[i], marker=markers[i], color = colors[i])

plt.legend(prop=font)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.xlabel('Date(May 2017)', fontdict=font)
plt.ylabel('Average detour distance for passengers (m)', fontdict=font)
plt.savefig('figure3_{}.png'.format('detour_distance')) 
plt.close()

plt.figure()  # 创建一个图形实例，方便同时多画几个图
for i in range(len(methods)):
    plt.plot(x,saved_ride_distance_dic[methods[i]]  , label=methods_name[i], marker=markers[i], color = colors[i])

plt.legend(prop=font)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.xlabel('Date(May 2017)', fontdict=font)
plt.ylabel('Total distance saving for drivers (km)', fontdict=font)
plt.savefig('figure3_{}.png'.format('saved_ride_distance')) 
plt.close()




plt.figure()  # 创建一个图形实例，方便同时多画几个图
for i in range(len(methods)):
    plt.plot(x,response_rate_dic[methods[i]] , label=methods_name[i], marker=markers[i], color = colors[i])

plt.legend(prop=font)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.xlabel('Date(May 2017)', fontdict=font)
plt.ylabel('Response rate', fontdict=font)
plt.savefig('figure3_{}.png'.format('response_rate')) 
plt.close()
plt.figure()  # 创建一个图形实例，方便同时多画几个图
for i in range(len(methods)):
    plt.plot(x,carpool_rate_dic[methods[i]] , label=methods_name[i],marker=markers[i],  color = colors[i])

plt.legend(prop=font)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.xlabel('Date(May 2017)', fontdict=font)
plt.ylabel('Ratio of successful matches', fontdict=font)
plt.savefig('figure3_{}.png'.format('carpool_rate')) 
plt.close()
