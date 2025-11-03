function [p, dp, chi2_red, xmin, ymin, dxmin, dymin] = fitquad(x, y, dy)
% FIT_QUADRATICO_ERRORI  Esegue un fit quadratico pesato ai dati sperimentali.
%
%   [p, dp, chi2_red] = fit_quadratico_errori(x, y, dy)
%
%   Input:
%       x  - vettore dei valori indipendenti
%       y  - vettore dei valori misurati
%       dy - incertezze (errori) sulle y
%
%   Output:
%       p        - parametri del fit [a, b, c] con y = a*x^2 + b*x + c
%       dp       - incertezze (deviazioni standard) sui parametri
%       chi2_red - chi-quadro ridotto del fit

    % Controlli di base
    if nargin < 3
        error('Servono tre vettori: x, y, dy');
    end
    if ~isequal(length(x), length(y), length(dy))
        error('x, y e dy devono avere la stessa lunghezza');
    end

    % Matrice del modello (design matrix)
    X = [x.^2, x, ones(size(x))];

    % Matrice di pesi (inverso degli errori al quadrato)
    W = diag(1 ./ (dy.^2));

    % Stima dei parametri (fit lineare pesato)
    Cov = inv(X' * W * X);
    p = Cov * (X' * W * y);

    % Deviazioni standard sui parametri
    dp = sqrt(diag(Cov));

    % Calcolo del chi-quadro
    y_fit = X * p;
    chi2 = sum(((y - y_fit) ./ dy).^2);
    chi2_red = chi2 / (length(y) - length(p));

    % === Calcolo del minimo ===
    a = p(1); b = p(2); c = p(3);
    xmin = -b / (2*a);
    ymin = c - b^2 / (4*a);

    % Propagazione errori (covarianza completa)
    % Var(xmin) = (∂xmin/∂a)^2 σ_a^2 + (∂xmin/∂b)^2 σ_b^2 + 2 (∂xmin/∂a)(∂xmin/∂b) cov(a,b)
    da_xmin = b / (2*a^2);
    db_xmin = -1 / (2*a);
    dxmin = sqrt( da_xmin^2 * Cov(1,1) + db_xmin^2 * Cov(2,2) + 2*da_xmin*db_xmin*Cov(1,2) );

    % Propagazione per ymin (in modo approssimato)
    % ymin = c - b^2/(4a)
    da_ymin = b^2 / (4*a^2);
    db_ymin = -b / (2*a);
    dc_ymin = 1;
    dymin = sqrt(...
        da_ymin^2 * Cov(1,1) + ...
        db_ymin^2 * Cov(2,2) + ...
        dc_ymin^2 * Cov(3,3) + ...
        2*(da_ymin*db_ymin*Cov(1,2) + da_ymin*dc_ymin*Cov(1,3) + db_ymin*dc_ymin*Cov(2,3)) ...
    );


    % Stampa dei risultati
    fprintf('Fit quadratico:\n');
    fprintf('  y = a*x^2 + b*x + c\n');
    fprintf('  a = %.6f ± %.6f\n', p(1), dp(1));
    fprintf('  b = %.6f ± %.6f\n', p(2), dp(2));
    fprintf('  c = %.6f ± %.6f\n', p(3), dp(3));
    fprintf('  chi^2 ridotto = %.3f\n', chi2_red);

    % Grafico principale e residui
    figure;

    % --- Grafico dati + fit ---
    subplot(2,1,1);
    hold on; grid on, box on;
    errorbar(x, y, dy, 'o', 'MarkerFaceColor', 'b');
    x_fit_plot = linspace(min(x), max(x), 200);
    y_fit_plot = p(1)*x_fit_plot.^2 + p(2)*x_fit_plot + p(3);
    plot(x_fit_plot, y_fit_plot, 'r-', 'LineWidth', 1.5);
    xlabel('x');
    ylabel('y');
    legend('Experimental data', 'Quadratic fit', 'Location', 'best');
    title('Fit quadratico con errori pesati');

    % --- Grafico residui ---
    subplot(2,1,2);
    residui = (y - y_fit);
    errorbar(x, residui, dy, 'o', 'MarkerFaceColor', 'b');
    yline(0, 'r--', 'LineWidth', 1);
    grid on;
    xlabel('x');
    ylabel('Residui (y - y_{fit})');
    title('Residui del fit');

end
