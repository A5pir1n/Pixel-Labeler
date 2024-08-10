#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stack>
#include <vector>
#include <unordered_set>

namespace py = pybind11;

class UndoRedoStack {
public:
    UndoRedoStack() : current_state(nullptr) {}

    void append(const py::dict& state) {
        if (current_state) {
            undo_stack.push(*current_state);
        }
        current_state = std::make_shared<py::dict>(state);
        while (!redo_stack.empty()) {
            redo_stack.pop();
        }
    }

    py::dict undo() {
        if (undo_stack.empty()) {
            throw std::runtime_error("Nothing to undo.");
        }
        redo_stack.push(*current_state);
        current_state = std::make_shared<py::dict>(undo_stack.top());
        undo_stack.pop();
        return *current_state;
    }

    py::dict redo() {
        if (redo_stack.empty()) {
            throw std::runtime_error("Nothing to redo.");
        }
        undo_stack.push(*current_state);
        current_state = std::make_shared<py::dict>(redo_stack.top());
        redo_stack.pop();
        return *current_state;
    }

    std::string print_history() const {
        std::ostringstream oss;
        oss << "Undo Stack:\n";
        auto temp_undo_stack = undo_stack;
        int index = 1;
        while (!temp_undo_stack.empty()) {
            const auto& state = temp_undo_stack.top();
            oss << "State " << index++ << ": "
                << len(state["foreground_pixels"].cast<std::unordered_set<std::tuple<int, int>>>()) << " foreground, "
                << len(state["background_pixels"].cast<std::unordered_set<std::tuple<int, int>>>()) << " background, "
                << len(state["unidentified_pixels"].cast<std::unordered_set<std::tuple<int, int>>>()) << " unidentified\n";
            temp_undo_stack.pop();
        }

        if (current_state) {
            const auto& state = *current_state;
            oss << "Current State: "
                << len(state["foreground_pixels"].cast<std::unordered_set<std::tuple<int, int>>>()) << " foreground, "
                << len(state["background_pixels"].cast<std::unordered_set<std::tuple<int, int>>>()) << " background, "
                << len(state["unidentified_pixels"].cast<std::unordered_set<std::tuple<int, int>>>()) << " unidentified\n";
        }

        oss << "Redo Stack:\n";
        auto temp_redo_stack = redo_stack;
        index = 1;
        while (!temp_redo_stack.empty()) {
            const auto& state = temp_redo_stack.top();
            oss << "Redo State " << index++ << ": "
                << len(state["foreground_pixels"].cast<std::unordered_set<std::tuple<int, int>>>()) << " foreground, "
                << len(state["background_pixels"].cast<std::unordered_set<std::tuple<int, int>>>()) << " background, "
                << len(state["unidentified_pixels"].cast<std::unordered_set<std::tuple<int, int>>>()) << " unidentified\n";
            temp_redo_stack.pop();
        }

        return oss.str();
    }

private:
    std::stack<py::dict> undo_stack;
    std::stack<py::dict> redo_stack;
    std::shared_ptr<py::dict> current_state;
};

PYBIND11_MODULE(undo_redo, m) {
    py::class_<UndoRedoStack>(m, "UndoRedoStack")
        .def(py::init<>())
        .def("append", &UndoRedoStack::append)
        .def("undo", &UndoRedoStack::undo)
        .def("redo", &UndoRedoStack::redo)
        .def("print_history", &UndoRedoStack::print_history);
}
