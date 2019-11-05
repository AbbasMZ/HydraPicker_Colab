import torch as T


class auxilary:
    Tparticle = T.empty(size=(2,1), dtype=T.int8)
    heads_training_only = T.empty(size=(2,1), dtype=T.int8)
    fine_tune = T.empty(size=(100, 1), dtype=T.int8)
    fine_tune_wgen = T.empty(size=(2, 1), dtype=T.int8)
    fine_tune_wfine = T.empty(size=(2, 1), dtype=T.int8)

    def heads_eval_mode(m):
        m.eval()
        for modules in [m.drop, m.sconv0, m.sconv1, m.sconv2, m.sconv3, m.out]:
            for module in modules:
                module.eval()
        return m

    def heads_train_mode(m):
        m.train()
        if auxilary.fine_tune[0] == 0:
            for modules in [m.drop, m.sconv0, m.sconv1, m.sconv2, m.sconv3, m.out]:
                for module in modules:
                    module.train()
            return m
        else:
            for modules in [m.drop, m.sconv0, m.sconv1, m.sconv2, m.sconv3, m.out]:
                for module in modules:
                    module.eval()
            for branch in auxilary.fine_tune[1:auxilary.fine_tune[0]]:
                for module in [m.drop[branch], m.sconv0[branch], m.sconv1[branch], m.sconv2[branch], m.sconv3[branch], m.out[branch]]:
                    module.train()
            if auxilary.fine_tune_wgen[0] == 1:
                branch = auxilary.fine_tune_wgen[1]
                for module in [m.drop[branch], m.sconv0[branch], m.sconv1[branch], m.sconv2[branch], m.sconv3[branch], m.out[branch]]:
                    module.train()
            if auxilary.fine_tune_wfine[0] == 1:
                branch = auxilary.fine_tune_wfine[1]
                for module in [m.drop[branch], m.sconv0[branch], m.sconv1[branch], m.sconv2[branch], m.sconv3[branch], m.out[branch]]:
                    module.train()
    # def heads_train_mode(m):
    #     m.train()
    #     if auxilary.auxilary.fine_tune[0] == 0:
    #         for modules in [m.drop, m.sconv0, m.sconv1, m.sconv2, m.sconv3, m.out]:
    #             for module in modules:
    #                 module.train()
    #         return m
    #     else:
    #         for modules in [m.drop, m.sconv0, m.sconv1, m.sconv2, m.sconv3, m.out]:
    #             for module in modules:
    #                 module.eval()
    #         for branch in auxilary.auxilary.fine_tune[1:auxilary.auxilary.fine_tune[0]]:
    #             for module in [m.drop[branch], m.sconv0[branch], m.sconv1[branch], m.sconv2[branch], m.sconv3[branch], m.out[branch]]:
    #                 module.train()
    #         if auxilary.auxilary.fine_tune_wgen[0] == 1:
    #             branch = auxilary.auxilary.fine_tune_wgen[1]
    #             for module in [m.drop[branch], m.sconv0[branch], m.sconv1[branch], m.sconv2[branch], m.sconv3[branch], m.out[branch]]:
    #                 module.train()
    #         if auxilary.auxilary.fine_tune_wfine[0] == 1:
    #             branch = auxilary.auxilary.fine_tune_wfine[1]
    #             for module in [m.drop[branch], m.sconv0[branch], m.sconv1[branch], m.sconv2[branch], m.sconv3[branch], m.out[branch]]:
    #                 module.train()