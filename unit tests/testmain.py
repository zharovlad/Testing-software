import unittest
import main
from numpy import array_equal


class TestCountRisk(unittest.TestCase):
    """ Тестирование подсчета матрицы рисков """

    def test_count_risk_matrix_1(self):
        start_matrix = [
            [2, 7, 8, 6],
            [2, 8, 7, 3],
            [4, 3, 4, 2]
        ]

        risk_matrix = [
            [2, 1, 0, 0],
            [2, 0, 1, 3],
            [0, 5, 4, 4]
        ]   

        result_matrix = main.count_risk(start_matrix)
        
        assert array_equal(result_matrix, risk_matrix) == True

    def test_count_risk_matrix_2(self):
        start_matrix = [
            [8, 8, 10, 15],
            [2, 4, 7, 9],
            [1, 1, 1, 1],
            [3, 3, 3, 3]
        ]

        risk_matrix = [
            [0, 0, 0, 0],
            [6, 4, 3, 6],
            [7, 7, 9, 14],
            [5, 5, 7, 12]
        ]   

        result_matrix = main.count_risk(start_matrix)
        
        assert array_equal(result_matrix, risk_matrix) == True


class TestProbability(unittest.TestCase):
    """ Тестирование решения, полученного из критерия, основанного на известных вероятностях условиях """
    def test_with_probaility_1(self):

        risk_matrix = [
            [2, 1, 0, 0],
            [2, 0, 1, 3],
            [0, 5, 4, 4]
        ]

        right_result_list = [0.7, 1.1, 3.5]

        probability_list = [0.2, 0.3, 0.4, 0.1]

        result_list = main.with_probability(risk_matrix, probability_list)

        assert array_equal(result_list, right_result_list) == True

    def test_with_probaility_2(self):

        risk_matrix = [
            [0, 0, 0, 0],
            [6, 4, 3, 6],
            [0, 0, 0, 0],
            [5, 5, 7, 12]
        ] 

        right_result_list = [0.0, 4.8, 0.0, 7.5]

        probability_list = [0.2, 0.3, 0.2, 0.3]

        result_list = main.with_probability(risk_matrix, probability_list)

        assert array_equal(result_list, right_result_list) == True


class TestWald(unittest.TestCase):
    """ Тестирование решения, полученного из критерия Вальда """
    
    def test_Wald_1(self):

        pay_matrix = [
            [2, 7, 8, 6],
            [2, 8, 7, 3],
            [4, 3, 4, 2]
        ]

        right_result_list = [2, 2, 2]

        result_list = main.Wald(pay_matrix)

        assert array_equal(result_list, right_result_list) == True
    
    def test_Wald_2(self):

        pay_matrix = [
            [8, 8, 10, 15],
            [2, 4, 7, 9],
            [1, 1, 1, 1],
            [3, 3, 3, 3]
        ]

        right_result_list = [8, 2, 1, 3]

        result_list = main.Wald(pay_matrix)

        assert array_equal(result_list, right_result_list) == True


class TestSavage(unittest.TestCase):
    """ Тестирование решения, полученного из критерия Сэвиджа """

    def test_Savage_1(self):

        risk_matrix = [
            [2, 1, 0, 0],
            [2, 0, 1, 3],
            [0, 5, 4, 4]
        ]

        right_result_list = [2, 3, 5]

        result_list = main.Savage(risk_matrix)

        assert array_equal(result_list, right_result_list) == True

    def test_Savage_2(self):

        risk_matrix = [
            [0, 0, 0, 0],
            [6, 4, 3, 6],
            [7, 7, 9, 14],
            [5, 5, 7, 12]
        ] 

        right_result_list = [0, 6, 14, 12]

        result_list = main.Savage(risk_matrix)

        assert array_equal(result_list, right_result_list) == True


class TestHurwitzMatrix(unittest.TestCase):
    """ Тестирование решения, полученного из критерия Гурвица, основанном на выигрыше """

    def test_Hurwitz_matrix_1(self):
        
        pay_matrix = [
            [2, 7, 8, 6],
            [2, 8, 7, 3],
            [4, 3, 4, 2]
        ]

        right_result_list = [5, 5, 3]

        result_list = main.Hurwitz_matrix(pay_matrix)

        assert array_equal(result_list, right_result_list) == True  

    def test_Hurwitz_matrix_2(self):

        pay_matrix = [
            [8, 8, 10, 15],
            [2, 4, 7, 9],
            [1, 1, 1, 1],
            [3, 3, 3, 3]
        ]

        right_result_list = [11.5, 5.5, 1, 3]

        result_list = main.Hurwitz_matrix(pay_matrix)

        assert array_equal(result_list, right_result_list) == True


class TestHurwitzRisk(unittest.TestCase):
    """ Тестирование решения, полученного из критерия Гурвица, основанном на риске """

    def test_Hurwitz_risk_1(self):

        risk_matrix = [
            [2, 1, 0, 0],
            [2, 0, 1, 3],
            [0, 5, 4, 4]
        ]

        right_result_list = [1, 1.5, 2.5]

        result_list = main.Hurwitz_risk(risk_matrix)

        assert array_equal(result_list, right_result_list) == True

    def test_Hurwitz_risk_2(self):

        risk_matrix = [
            [0, 0, 0, 0],
            [6, 4, 3, 6],
            [7, 7, 9, 14],
            [5, 5, 7, 12]
        ] 

        right_result_list = [0, 4.5, 10.5, 8.5]

        result_list = main.Hurwitz_risk(risk_matrix)

        assert array_equal(result_list, right_result_list) == True


if __name__ == "__main__":
    unittest.main()