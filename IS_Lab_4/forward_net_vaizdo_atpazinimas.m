close all
clear all
clc

pavadinimas = 'train_data.png';
pozymiai_tinklo_mokymui = pozymiai_raidems_atpazinti(pavadinimas, 8);

P = cell2mat(pozymiai_tinklo_mokymui);
T = [eye(10), eye(10), eye(10), eye(10), eye(10), eye(10), eye(10), eye(10)];

tinklas_ffn = feedforwardnet([20, 10]);
tinklas_ffn = train(tinklas_ffn, P, T);

%% Extract features of the test image
pavadinimas = 'test_kada.png';
pozymiai_patikrai = pozymiai_raidems_atpazinti(pavadinimas, 1);

P2 = cell2mat(pozymiai_patikrai);
Y = tinklas_ffn(P2);

[~, idx] = max(Y);
symbols = '0123456789';
atsakymas = symbols(idx);

disp(atsakymas);
figure(9), text(0.1,0.5,atsakymas,'FontSize',38), axis off